#include "EmissionProgram.h"
#include "inter/Emit/Emit.h"

#include "ged.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <variant>

using namespace mlir;
using namespace inter::xemachine;

namespace inter::detail {
namespace {

constexpr uint32_t nativeInstructionSize = 16;

class GedEncoder {
public:
  GedEncoder(ModuleOp moduleOp, llvm::raw_ostream &output, GED_MODEL model)
      : moduleOp(moduleOp), output(output), model(model) {}

  LogicalResult encode(const EmissionProgram &program) {
    if (failed(layout(program)))
      return failure();

    llvm::SmallVector<char> binary;
    binary.reserve(binarySize);
    currentPc = 0;
    for (const EmissionItem &item : program.items) {
      if (std::holds_alternative<Label>(item))
        continue;

      ged_ins_t instruction;
      if (failed(std::visit(
              [&](const auto &value) { return encodeItem(value, instruction); },
              item)))
        return failure();

      std::array<unsigned char, nativeInstructionSize> bytes{};
      if (failed(check(
              GED_EncodeIns(&instruction, GED_INS_TYPE_NATIVE, bytes.data()),
              "GED_EncodeIns")))
        return failure();
      const char *begin = reinterpret_cast<const char *>(bytes.data());
      binary.append(begin, begin + bytes.size());
      currentPc += nativeInstructionSize;
    }

    output.write(binary.data(), binary.size());
    return success();
  }

  std::optional<uint32_t> getLabelOffset(uint32_t label) const {
    auto iterator = labelOffsets.find(label);
    if (iterator == labelOffsets.end())
      return std::nullopt;
    return iterator->second;
  }

private:
  LogicalResult layout(const EmissionProgram &program) {
    uint64_t pc = 0;
    for (const EmissionItem &item : program.items) {
      if (const Label *label = std::get_if<Label>(&item)) {
        if (pc > std::numeric_limits<uint32_t>::max()) {
          moduleOp.emitError(
              "encoded kernel exceeds the 32-bit GED address space");
          return failure();
        }
        if (!labelOffsets.try_emplace(label->id, static_cast<uint32_t>(pc))
                 .second) {
          moduleOp.emitError() << "duplicate emission label L" << label->id;
          return failure();
        }
        continue;
      }
      pc += nativeInstructionSize;
    }
    if (pc > std::numeric_limits<uint32_t>::max()) {
      moduleOp.emitError("encoded kernel exceeds the 32-bit GED address space");
      return failure();
    }
    binarySize = static_cast<uint32_t>(pc);
    return success();
  }

  LogicalResult check(GED_RETURN_VALUE result, llvm::StringRef operation) {
    if (result == GED_RETURN_VALUE_SUCCESS)
      return success();
    moduleOp.emitError() << operation << " failed at byte " << currentPc << ": "
                         << GED_GetReturnValueString(result);
    return failure();
  }

#define RETURN_IF_GED_ERROR(call)                                              \
  do {                                                                         \
    if (failed(check((call), #call)))                                          \
      return failure();                                                        \
  } while (false)

  static GED_OPCODE getOpcode(AluOpcode opcode) {
    switch (opcode) {
    case AluOpcode::mov:
      return GED_OPCODE_mov;
    case AluOpcode::add:
      return GED_OPCODE_add;
    case AluOpcode::shl:
      return GED_OPCODE_shl;
    case AluOpcode::shr:
      return GED_OPCODE_shr;
    case AluOpcode::and_:
      return GED_OPCODE_and;
    case AluOpcode::or_:
      return GED_OPCODE_or;
    case AluOpcode::add3:
      return GED_OPCODE_add3;
    case AluOpcode::csel:
      return GED_OPCODE_csel;
    case AluOpcode::mul:
      return GED_OPCODE_mul;
    }
    llvm_unreachable("unknown ALU opcode");
  }

  static GED_DATA_TYPE getDataType(DataType type) {
    switch (type) {
    case DataType::ub:
      return GED_DATA_TYPE_ub;
    case DataType::uw:
      return GED_DATA_TYPE_uw;
    case DataType::ud:
      return GED_DATA_TYPE_ud;
    case DataType::q:
      return GED_DATA_TYPE_q;
    case DataType::f:
      return GED_DATA_TYPE_f;
    }
    llvm_unreachable("unknown data type");
  }

  static GED_DATA_TYPE getDataType(DataType type, bool isSigned) {
    if (!isSigned)
      return getDataType(type);
    switch (type) {
    case DataType::ub:
      return GED_DATA_TYPE_b;
    case DataType::uw:
      return GED_DATA_TYPE_w;
    case DataType::ud:
      return GED_DATA_TYPE_d;
    case DataType::q:
      return GED_DATA_TYPE_q;
    case DataType::f:
      return GED_DATA_TYPE_f;
    }
    llvm_unreachable("unknown signed data type");
  }

  static uint32_t getTypeBytes(DataType type) {
    switch (type) {
    case DataType::ub:
      return 1;
    case DataType::uw:
      return 2;
    case DataType::ud:
    case DataType::f:
      return 4;
    case DataType::q:
      return 8;
    }
    llvm_unreachable("unknown data type");
  }

  LogicalResult setExecution(ged_ins_t &instruction,
                             const ExecutionInfo &execution) {
    RETURN_IF_GED_ERROR(GED_SetExecSize(&instruction, execution.size));

    GED_CHANNEL_OFFSET channelOffset;
    switch (execution.maskOffset) {
    case 0:
      channelOffset = GED_CHANNEL_OFFSET_M0;
      break;
    case 8:
      channelOffset = GED_CHANNEL_OFFSET_M8;
      break;
    case 16:
      channelOffset = GED_CHANNEL_OFFSET_M16;
      break;
    case 24:
      channelOffset = GED_CHANNEL_OFFSET_M24;
      break;
    default:
      moduleOp.emitError() << "unsupported mask offset "
                           << execution.maskOffset;
      return failure();
    }
    RETURN_IF_GED_ERROR(GED_SetChannelOffset(&instruction, channelOffset));
    RETURN_IF_GED_ERROR(
        GED_SetMaskCtrl(&instruction, execution.noMask ? GED_MASK_CTRL_NoMask
                                                       : GED_MASK_CTRL_Normal));
    return success();
  }

  LogicalResult setPredicate(ged_ins_t &instruction,
                             const std::optional<Predicate> &predicate) {
    if (!predicate) {
      RETURN_IF_GED_ERROR(GED_SetPredCtrl(&instruction, GED_PRED_CTRL_Normal));
      return success();
    }
    if (predicate->flag.file != ARFFile::f) {
      moduleOp.emitError("GED predicates require a flag register");
      return failure();
    }
    RETURN_IF_GED_ERROR(
        GED_SetPredCtrl(&instruction, GED_PRED_CTRL_Sequential));
    RETURN_IF_GED_ERROR(
        GED_SetPredInv(&instruction, predicate->inverse ? GED_PRED_INV_Invert
                                                        : GED_PRED_INV_Normal));
    RETURN_IF_GED_ERROR(
        GED_SetFlagRegNum(&instruction, predicate->flag.number));
    RETURN_IF_GED_ERROR(
        GED_SetFlagSubRegNum(&instruction, predicate->flag.sub));
    return success();
  }

  FailureOr<uint32_t> getSwsb(const SwsbInfo &swsb, bool isSend) {
    if (swsb.distance > 7) {
      moduleOp.emitError() << "SWSB distance exceeds 7 at byte " << currentPc;
      return failure();
    }
    if (swsb.token > 31) {
      moduleOp.emitError() << "SWSB token exceeds 31 at byte " << currentPc;
      return failure();
    }
    if (swsb.distance >= 0 && swsb.pipe == DistancePipe::none) {
      moduleOp.emitError() << "SWSB distance has no pipe at byte " << currentPc;
      return failure();
    }

    if (swsb.distance >= 0 && swsb.token >= 0) {
      if (!isSend) {
        moduleOp.emitError() << "combined SWSB distance and token requires a "
                                "send at byte "
                             << currentPc;
        return failure();
      }
      if (swsb.pipe == DistancePipe::floating) {
        moduleOp.emitError() << "combined floating SWSB distance and token is "
                                "not produced for sends at byte "
                             << currentPc;
        return failure();
      }
      uint32_t mode = swsb.pipe == DistancePipe::all ? 0x100 : 0x300;
      return mode | (static_cast<uint32_t>(swsb.distance) << 5) |
             static_cast<uint32_t>(swsb.token);
    }
    if (swsb.distance >= 0) {
      uint32_t pipe = swsb.pipe == DistancePipe::all        ? 0x08
                      : swsb.pipe == DistancePipe::floating ? 0x10
                                                            : 0x18;
      return pipe | static_cast<uint32_t>(swsb.distance);
    }
    if (swsb.token >= 0) {
      uint32_t mode = swsb.tokenMode == TokenMode::destination ? 0x80
                      : swsb.tokenMode == TokenMode::source    ? 0xA0
                                                               : 0xC0;
      return mode | static_cast<uint32_t>(swsb.token);
    }
    return 0;
  }

  LogicalResult setOptions(ged_ins_t &instruction, const SwsbInfo &swsb,
                           bool isSend,
                           std::optional<uint32_t> rawSwsb = std::nullopt) {
    RETURN_IF_GED_ERROR(GED_SetDebugCtrl(&instruction, GED_DEBUG_CTRL_Normal));
    if (!isSend)
      RETURN_IF_GED_ERROR(
          GED_SetThreadCtrl(&instruction, GED_THREAD_CTRL_Normal));
    uint32_t encodedSwsb;
    if (rawSwsb) {
      encodedSwsb = *rawSwsb;
    } else {
      FailureOr<uint32_t> encoded = getSwsb(swsb, isSend);
      if (failed(encoded))
        return failure();
      encodedSwsb = *encoded;
    }
    RETURN_IF_GED_ERROR(GED_SetSWSB(&instruction, encodedSwsb));
    return success();
  }

  LogicalResult
  initialize(ged_ins_t &instruction, GED_OPCODE opcode,
             const ExecutionInfo &execution,
             const std::optional<Predicate> &predicate = std::nullopt) {
    RETURN_IF_GED_ERROR(GED_InitEmptyIns(model, &instruction, opcode));
    if (failed(setExecution(instruction, execution)))
      return failure();
    return setPredicate(instruction, predicate);
  }

  FailureOr<uint32_t> getArfRegisterNumber(ArfReference reference) {
    GED_ARCH_REG archRegister = GED_ARCH_REG_INVALID;
    uint32_t number = reference.number;
    switch (reference.file) {
    case ARFFile::a0:
      archRegister = GED_ARCH_REG_a0;
      break;
    case ARFFile::f:
      archRegister = GED_ARCH_REG_f;
      break;
    case ARFFile::acc:
      archRegister = GED_ARCH_REG_acc;
      break;
    case ARFFile::mme:
      archRegister = GED_ARCH_REG_acc;
      number += 8;
      break;
    case ARFFile::sr:
      archRegister = GED_ARCH_REG_sr0;
      break;
    case ARFFile::cr:
      archRegister = GED_ARCH_REG_cr0;
      break;
    case ARFFile::n:
      archRegister = GED_ARCH_REG_n;
      break;
    case ARFFile::ip:
      archRegister = GED_ARCH_REG_ip;
      break;
    case ARFFile::tdr:
      archRegister = GED_ARCH_REG_tdr;
      break;
    case ARFFile::tm:
      archRegister = GED_ARCH_REG_tm0;
      break;
    case ARFFile::fc:
      archRegister = GED_ARCH_REG_fc;
      break;
    case ARFFile::dbg:
      archRegister = GED_ARCH_REG_dbg0;
      break;
    }

    uint32_t encoded = 0;
    if (failed(check(GED_SetArchReg(&encoded, model, archRegister),
                     "GED_SetArchReg")) ||
        failed(check(GED_SetArchRegNum(&encoded, model, number),
                     "GED_SetArchRegNum")))
      return failure();
    return encoded;
  }

  LogicalResult getRegister(const RegisterReference &reference, DataType type,
                            GED_REG_FILE &file, uint32_t &number,
                            uint32_t &subRegister) {
    if (const auto *grf = std::get_if<GrfReference>(&reference)) {
      if (grf->number < 0 || grf->sub < 0) {
        moduleOp.emitError("GED emission requires physical GRF references");
        return failure();
      }
      file = GED_REG_FILE_GRF;
      number = grf->number;
      subRegister = grf->sub * getTypeBytes(type);
      return success();
    }

    const ArfReference &arf = std::get<ArfReference>(reference);
    if (arf.number < 0 || arf.sub < 0) {
      moduleOp.emitError("GED emission requires physical ARF references");
      return failure();
    }
    FailureOr<uint32_t> encoded = getArfRegisterNumber(arf);
    if (failed(encoded))
      return failure();
    file = GED_REG_FILE_ARF;
    number = *encoded;
    subRegister = arf.sub * getTypeBytes(type);
    return success();
  }

  static uint64_t negateImmediate(uint64_t value, DataType type) {
    if (type == DataType::f)
      return value ^ (uint64_t{1} << 31);
    uint32_t bits = getTypeBytes(type) * 8;
    if (bits == 64)
      return 0 - value;
    uint64_t mask = (uint64_t{1} << bits) - 1;
    return (0 - value) & mask;
  }

  LogicalResult setBasicSource(ged_ins_t &instruction, uint32_t index,
                               const SourceOperand &source) {
    GED_DATA_TYPE type = getDataType(source.type, source.isSigned);
    if (const auto *immediate = std::get_if<Immediate>(&source.value)) {
      GED_DATA_TYPE immediateType = type;
      if (index == 0) {
        RETURN_IF_GED_ERROR(GED_SetSrc0RegFile(&instruction, GED_REG_FILE_IMM));
        RETURN_IF_GED_ERROR(GED_SetSrc0DataType(&instruction, immediateType));
      } else {
        RETURN_IF_GED_ERROR(GED_SetSrc1RegFile(&instruction, GED_REG_FILE_IMM));
        RETURN_IF_GED_ERROR(GED_SetSrc1DataType(&instruction, immediateType));
      }
      uint64_t value = source.negate
                           ? negateImmediate(immediate->value, immediate->type)
                           : immediate->value;
      unsigned bits = getTypeBytes(source.type) * 8;
      if (bits < 64)
        value &= (uint64_t{1} << bits) - 1;
      RETURN_IF_GED_ERROR(GED_SetImm(&instruction, value));
      return success();
    }

    RegisterReference reference;
    if (const auto *grf = std::get_if<GrfReference>(&source.value))
      reference = *grf;
    else
      reference = std::get<ArfReference>(source.value);
    GED_REG_FILE file;
    uint32_t number;
    uint32_t subRegister;
    if (failed(getRegister(reference, source.type, file, number, subRegister)))
      return failure();
    GED_SRC_MOD modifier =
        source.negate ? GED_SRC_MOD_Negative : GED_SRC_MOD_Normal;
    if (index == 0) {
      RETURN_IF_GED_ERROR(GED_SetSrc0RegFile(&instruction, file));
      RETURN_IF_GED_ERROR(GED_SetSrc0SrcMod(&instruction, modifier));
      RETURN_IF_GED_ERROR(GED_SetSrc0DataType(&instruction, type));
      RETURN_IF_GED_ERROR(
          GED_SetSrc0AddrMode(&instruction, GED_ADDR_MODE_Direct));
      RETURN_IF_GED_ERROR(GED_SetSrc0RegNum(&instruction, number));
      RETURN_IF_GED_ERROR(GED_SetSrc0SubRegNum(&instruction, subRegister));
      RETURN_IF_GED_ERROR(
          GED_SetSrc0VertStride(&instruction, source.region.vstride));
      RETURN_IF_GED_ERROR(GED_SetSrc0Width(&instruction, source.region.width));
      RETURN_IF_GED_ERROR(
          GED_SetSrc0HorzStride(&instruction, source.region.hstride));
    } else {
      RETURN_IF_GED_ERROR(GED_SetSrc1RegFile(&instruction, file));
      RETURN_IF_GED_ERROR(GED_SetSrc1SrcMod(&instruction, modifier));
      RETURN_IF_GED_ERROR(GED_SetSrc1DataType(&instruction, type));
      RETURN_IF_GED_ERROR(
          GED_SetSrc1AddrMode(&instruction, GED_ADDR_MODE_Direct));
      RETURN_IF_GED_ERROR(GED_SetSrc1RegNum(&instruction, number));
      RETURN_IF_GED_ERROR(GED_SetSrc1SubRegNum(&instruction, subRegister));
      RETURN_IF_GED_ERROR(
          GED_SetSrc1VertStride(&instruction, source.region.vstride));
      RETURN_IF_GED_ERROR(GED_SetSrc1Width(&instruction, source.region.width));
      RETURN_IF_GED_ERROR(
          GED_SetSrc1HorzStride(&instruction, source.region.hstride));
    }
    return success();
  }

  LogicalResult
  setBasicDestination(ged_ins_t &instruction,
                      const std::optional<Destination> &destination,
                      DataType type, bool supportsSaturation) {
    GED_REG_FILE file = GED_REG_FILE_ARF;
    uint32_t number = 0;
    uint32_t subRegister = 0;
    uint32_t horizontalStride = 1;
    if (destination) {
      if (failed(
              getRegister(destination->value, type, file, number, subRegister)))
        return failure();
      horizontalStride = destination->hstride;
    }
    RETURN_IF_GED_ERROR(GED_SetDstRegFile(&instruction, file));
    RETURN_IF_GED_ERROR(GED_SetDstAddrMode(&instruction, GED_ADDR_MODE_Direct));
    RETURN_IF_GED_ERROR(GED_SetDstDataType(&instruction, getDataType(type)));
    if (supportsSaturation)
      RETURN_IF_GED_ERROR(GED_SetSaturate(&instruction, GED_SATURATE_Normal));
    RETURN_IF_GED_ERROR(GED_SetDstRegNum(&instruction, number));
    RETURN_IF_GED_ERROR(GED_SetDstSubRegNum(&instruction, subRegister));
    RETURN_IF_GED_ERROR(GED_SetDstHorzStride(&instruction, horizontalStride));
    return success();
  }

  LogicalResult setTernarySource(ged_ins_t &instruction, uint32_t index,
                                 const SourceOperand &source) {
    if (const auto *immediate = std::get_if<Immediate>(&source.value)) {
      if (index == 1) {
        moduleOp.emitError("GED ternary source 1 cannot be immediate");
        return failure();
      }
      GED_DATA_TYPE type = getDataType(immediate->type);
      if (index == 0) {
        RETURN_IF_GED_ERROR(GED_SetSrc0DataType(&instruction, type));
        RETURN_IF_GED_ERROR(GED_SetSrc0RegFile(&instruction, GED_REG_FILE_IMM));
        uint64_t value =
            source.negate ? negateImmediate(immediate->value, immediate->type)
                          : immediate->value;
        RETURN_IF_GED_ERROR(GED_SetSrc0TernaryImm(&instruction, value));
      } else {
        RETURN_IF_GED_ERROR(GED_SetSrc2DataType(&instruction, type));
        RETURN_IF_GED_ERROR(GED_SetSrc2RegFile(&instruction, GED_REG_FILE_IMM));
        uint64_t value =
            source.negate ? negateImmediate(immediate->value, immediate->type)
                          : immediate->value;
        RETURN_IF_GED_ERROR(GED_SetSrc2TernaryImm(&instruction, value));
      }
      return success();
    }

    RegisterReference reference;
    if (const auto *grf = std::get_if<GrfReference>(&source.value))
      reference = *grf;
    else
      reference = std::get<ArfReference>(source.value);
    GED_REG_FILE file;
    uint32_t number;
    uint32_t subRegister;
    if (failed(getRegister(reference, source.type, file, number, subRegister)))
      return failure();
    GED_DATA_TYPE type = getDataType(source.type, source.isSigned);
    GED_SRC_MOD modifier =
        source.negate ? GED_SRC_MOD_Negative : GED_SRC_MOD_Normal;
    if (index == 0) {
      RETURN_IF_GED_ERROR(GED_SetSrc0DataType(&instruction, type));
      RETURN_IF_GED_ERROR(GED_SetSrc0RegFile(&instruction, file));
      RETURN_IF_GED_ERROR(GED_SetSrc0SrcMod(&instruction, modifier));
      RETURN_IF_GED_ERROR(
          GED_SetSrc0VertStride(&instruction, source.region.vstride));
      RETURN_IF_GED_ERROR(
          GED_SetSrc0HorzStride(&instruction, source.region.hstride));
      RETURN_IF_GED_ERROR(GED_SetSrc0RegNum(&instruction, number));
      RETURN_IF_GED_ERROR(GED_SetSrc0SubRegNum(&instruction, subRegister));
    } else if (index == 1) {
      RETURN_IF_GED_ERROR(GED_SetSrc1DataType(&instruction, type));
      RETURN_IF_GED_ERROR(GED_SetSrc1RegFile(&instruction, file));
      RETURN_IF_GED_ERROR(GED_SetSrc1SrcMod(&instruction, modifier));
      RETURN_IF_GED_ERROR(
          GED_SetSrc1VertStride(&instruction, source.region.vstride));
      RETURN_IF_GED_ERROR(
          GED_SetSrc1HorzStride(&instruction, source.region.hstride));
      RETURN_IF_GED_ERROR(GED_SetSrc1RegNum(&instruction, number));
      RETURN_IF_GED_ERROR(GED_SetSrc1SubRegNum(&instruction, subRegister));
    } else {
      RETURN_IF_GED_ERROR(GED_SetSrc2DataType(&instruction, type));
      RETURN_IF_GED_ERROR(GED_SetSrc2RegFile(&instruction, file));
      RETURN_IF_GED_ERROR(GED_SetSrc2SrcMod(&instruction, modifier));
      RETURN_IF_GED_ERROR(
          GED_SetSrc2HorzStride(&instruction, source.region.hstride));
      RETURN_IF_GED_ERROR(GED_SetSrc2RegNum(&instruction, number));
      RETURN_IF_GED_ERROR(GED_SetSrc2SubRegNum(&instruction, subRegister));
    }
    return success();
  }

  LogicalResult encodeItem(const Label &, ged_ins_t &) {
    llvm_unreachable("labels do not encode instructions");
  }

  LogicalResult encodeItem(const AluInstruction &value,
                           ged_ins_t &instruction) {
    if (failed(
            initialize(instruction, getOpcode(value.opcode), value.execution)))
      return failure();

    bool immediate64Source0 =
        !value.sources.empty() &&
        std::holds_alternative<Immediate>(value.sources.front().value) &&
        std::get<Immediate>(value.sources.front().value).type == DataType::q;
    if (!immediate64Source0)
      RETURN_IF_GED_ERROR(
          GED_SetCondModifier(&instruction, GED_COND_MODIFIER_Normal));

    bool ternary =
        value.opcode == AluOpcode::add3 || value.opcode == AluOpcode::csel;
    if (ternary) {
      if (value.sources.size() != 3 || !value.destination) {
        moduleOp.emitError("ternary ALU instruction requires one destination "
                           "and three sources");
        return failure();
      }
      GED_EXECUTION_DATA_TYPE executionType =
          value.sources.front().type == DataType::f
              ? GED_EXECUTION_DATA_TYPE_Float
              : GED_EXECUTION_DATA_TYPE_Integer;
      RETURN_IF_GED_ERROR(
          GED_SetExecutionDataType(&instruction, executionType));

      GED_REG_FILE file;
      uint32_t number;
      uint32_t subRegister;
      if (failed(getRegister(value.destination->value, value.destinationType,
                             file, number, subRegister)))
        return failure();
      RETURN_IF_GED_ERROR(GED_SetSaturate(&instruction, GED_SATURATE_Normal));
      RETURN_IF_GED_ERROR(GED_SetDstDataType(
          &instruction,
          getDataType(value.destinationType, value.destinationSigned)));
      RETURN_IF_GED_ERROR(GED_SetDstRegFile(&instruction, file));
      RETURN_IF_GED_ERROR(GED_SetDstRegNum(&instruction, number));
      RETURN_IF_GED_ERROR(GED_SetDstSubRegNum(&instruction, subRegister));
      RETURN_IF_GED_ERROR(
          GED_SetDstHorzStride(&instruction, value.destination->hstride));
      for (auto [index, source] : llvm::enumerate(value.sources))
        if (failed(setTernarySource(instruction, index, source)))
          return failure();
      if (value.opcode == AluOpcode::csel) {
        if (!value.condition || !value.flag || value.flag->file != ARFFile::f) {
          moduleOp.emitError("csel requires a condition and flag destination");
          return failure();
        }
        RETURN_IF_GED_ERROR(
            GED_SetCondModifier(&instruction, getCondition(*value.condition)));
        RETURN_IF_GED_ERROR(
            GED_SetFlagRegNum(&instruction, value.flag->number));
        RETURN_IF_GED_ERROR(
            GED_SetFlagSubRegNum(&instruction, value.flag->sub));
      }
    } else {
      uint32_t expectedSources = value.opcode == AluOpcode::mov ? 1 : 2;
      if (value.sources.size() != expectedSources) {
        moduleOp.emitError("incorrect source count for GED ALU instruction");
        return failure();
      }
      uint32_t immediateCount =
          llvm::count_if(value.sources, [](const auto &source) {
            return std::holds_alternative<Immediate>(source.value);
          });
      if (immediateCount > 1) {
        moduleOp.emitError(
            "basic instructions cannot encode two immediate sources");
        return failure();
      }
      if (expectedSources == 2 &&
          failed(setBasicSource(instruction, 1, value.sources[1])))
        return failure();
      if (failed(setBasicSource(instruction, 0, value.sources[0])))
        return failure();
      bool supportsSaturation =
          value.opcode != AluOpcode::and_ && value.opcode != AluOpcode::or_;
      if (failed(setBasicDestination(instruction, value.destination,
                                     value.destinationType,
                                     supportsSaturation)))
        return failure();
    }
    return setOptions(instruction, value.swsb, false);
  }

  static GED_COND_MODIFIER getCondition(CondModifier condition) {
    switch (condition) {
    case CondModifier::eq:
      return GED_COND_MODIFIER_z;
    case CondModifier::ne:
      return GED_COND_MODIFIER_nz;
    case CondModifier::lt:
      return GED_COND_MODIFIER_l;
    case CondModifier::le:
      return GED_COND_MODIFIER_le;
    case CondModifier::gt:
      return GED_COND_MODIFIER_g;
    case CondModifier::ge:
      return GED_COND_MODIFIER_ge;
    }
    llvm_unreachable("unknown compare condition");
  }

  LogicalResult encodeItem(const CompareInstruction &value,
                           ged_ins_t &instruction) {
    if (value.sources.size() != 2 || value.flag.file != ARFFile::f) {
      moduleOp.emitError("cmp requires two sources and a flag destination");
      return failure();
    }
    if (failed(initialize(instruction, GED_OPCODE_cmp, value.execution)))
      return failure();
    RETURN_IF_GED_ERROR(
        GED_SetCondModifier(&instruction, getCondition(value.condition)));
    RETURN_IF_GED_ERROR(GED_SetPredInv(&instruction, GED_PRED_INV_Normal));
    RETURN_IF_GED_ERROR(GED_SetFlagRegNum(&instruction, value.flag.number));
    RETURN_IF_GED_ERROR(GED_SetFlagSubRegNum(&instruction, value.flag.sub));
    if (failed(setBasicSource(instruction, 1, value.sources[1])) ||
        failed(setBasicSource(instruction, 0, value.sources[0])) ||
        failed(setBasicDestination(instruction, std::nullopt, value.dataType,
                                   false)))
      return failure();
    return setOptions(instruction, value.swsb, false);
  }

  LogicalResult encodeItem(const DpasInstruction &value,
                           ged_ins_t &instruction) {
    auto precision = [](DpasPrecision value) {
      return value == DpasPrecision::F16 ? GED_PRECISION_f16
                                         : GED_PRECISION_bf16;
    };
    auto verifyRegister = [&](GrfReference reference, StringRef name) {
      if (reference.number < 0 || reference.sub != 0) {
        moduleOp.emitError()
            << "DPAS " << name << " must be a physical GRF-aligned register";
        return failure();
      }
      return success();
    };
    if (failed(verifyRegister(value.destination, "destination")) ||
        failed(verifyRegister(value.accumulator, "accumulator")) ||
        failed(verifyRegister(value.sourceB, "B source")) ||
        failed(verifyRegister(value.sourceA, "A source")))
      return failure();
    if (failed(initialize(instruction, GED_OPCODE_dpas, value.execution)))
      return failure();
    RETURN_IF_GED_ERROR(
        GED_SetExecutionDataType(&instruction, GED_EXECUTION_DATA_TYPE_Float));
    RETURN_IF_GED_ERROR(
        GED_SetSystolicDepth(&instruction, value.systolicDepth));
    RETURN_IF_GED_ERROR(GED_SetRepeatCount(&instruction, value.repeatCount));
    RETURN_IF_GED_ERROR(GED_SetSrc0DataType(&instruction, GED_DATA_TYPE_f));
    RETURN_IF_GED_ERROR(
        GED_SetSrc1Precision(&instruction, precision(value.bPrecision)));
    RETURN_IF_GED_ERROR(
        GED_SetSrc2Precision(&instruction, precision(value.aPrecision)));
    RETURN_IF_GED_ERROR(GED_SetDstDataType(&instruction, GED_DATA_TYPE_f));
    RETURN_IF_GED_ERROR(GED_SetDstRegFile(&instruction, GED_REG_FILE_GRF));
    RETURN_IF_GED_ERROR(
        GED_SetDstRegNum(&instruction, value.destination.number));
    RETURN_IF_GED_ERROR(GED_SetDstSubRegNum(&instruction, 0));
    RETURN_IF_GED_ERROR(GED_SetDstHorzStride(&instruction, 1));
    RETURN_IF_GED_ERROR(GED_SetSrc0RegFile(&instruction, GED_REG_FILE_GRF));
    RETURN_IF_GED_ERROR(
        GED_SetSrc0RegNum(&instruction, value.accumulator.number));
    RETURN_IF_GED_ERROR(GED_SetSrc0SubRegNum(&instruction, 0));
    RETURN_IF_GED_ERROR(GED_SetSrc1RegFile(&instruction, GED_REG_FILE_GRF));
    RETURN_IF_GED_ERROR(GED_SetSrc1RegNum(&instruction, value.sourceB.number));
    RETURN_IF_GED_ERROR(GED_SetSrc1SubRegNum(&instruction, 0));
    RETURN_IF_GED_ERROR(GED_SetSrc2RegFile(&instruction, GED_REG_FILE_GRF));
    RETURN_IF_GED_ERROR(GED_SetSrc2RegNum(&instruction, value.sourceA.number));
    RETURN_IF_GED_ERROR(GED_SetSrc2SubRegNum(&instruction, 0));
    return setOptions(instruction, value.swsb, false);
  }

  static GED_SFID getSfid(SendFn function) {
    switch (function) {
    case SendFn::ugm:
      return GED_SFID_UGM;
    case SendFn::gtwy:
      return GED_SFID_GATEWAY;
    case SendFn::tgm:
      return GED_SFID_TGM;
    case SendFn::slm:
      return GED_SFID_SLM;
    }
    llvm_unreachable("unknown send function");
  }

  LogicalResult encodeItem(const SendInstruction &value,
                           ged_ins_t &instruction) {
    if (value.address.number < 0 ||
        (value.destination && value.destination->number < 0) ||
        (value.data && value.data->number < 0)) {
      moduleOp.emitError("GED emission requires physical send registers");
      return failure();
    }
    if (failed(initialize(instruction, GED_OPCODE_send, value.execution)))
      return failure();
    RETURN_IF_GED_ERROR(GED_SetSFID(&instruction, getSfid(value.function)));

    RETURN_IF_GED_ERROR(GED_SetDstRegFile(
        &instruction, value.destination ? GED_REG_FILE_GRF : GED_REG_FILE_ARF));
    RETURN_IF_GED_ERROR(GED_SetDstRegNum(
        &instruction, value.destination ? value.destination->number : 0));
    if (value.destination && value.destination->sub != 0) {
      moduleOp.emitError("send destination must be GRF-aligned");
      return failure();
    }

    if (value.address.sub != 0) {
      moduleOp.emitError("send address must be GRF-aligned");
      return failure();
    }
    RETURN_IF_GED_ERROR(GED_SetSrc0RegFile(&instruction, GED_REG_FILE_GRF));
    RETURN_IF_GED_ERROR(GED_SetSrc0RegNum(&instruction, value.address.number));

    RETURN_IF_GED_ERROR(GED_SetSrc1RegFile(
        &instruction, value.data ? GED_REG_FILE_GRF : GED_REG_FILE_ARF));
    uint32_t sourceLength = 0;
    if (value.data) {
      if (value.data->sub != 0 || value.data->widthDwords % 16 != 0) {
        moduleOp.emitError(
            "send data must contain whole, GRF-aligned registers");
        return failure();
      }
      sourceLength = value.data->widthDwords / 16;
      RETURN_IF_GED_ERROR(GED_SetSrc1RegNum(&instruction, value.data->number));
    }

    if (const auto *immediate = std::get_if<uint32_t>(&value.exdesc)) {
      RETURN_IF_GED_ERROR(GED_SetExDescRegFile(&instruction, GED_REG_FILE_IMM));
      RETURN_IF_GED_ERROR(GED_SetExMsgDescImm(&instruction, *immediate));
    } else {
      const ExtendedDescriptorReference &exdesc =
          std::get<ExtendedDescriptorReference>(value.exdesc);
      if (exdesc.base.file != ARFFile::a0 || exdesc.base.number != 0 ||
          exdesc.base.sub != 2) {
        moduleOp.emitError("register exdesc must use a0.2");
        return failure();
      }
      RETURN_IF_GED_ERROR(GED_SetExDescRegFile(&instruction, GED_REG_FILE_ARF));
      RETURN_IF_GED_ERROR(GED_SetExMsgDescImm(&instruction, exdesc.immediate));
      RETURN_IF_GED_ERROR(
          GED_SetExDescAddrSubRegNum(&instruction, 2 * exdesc.base.sub));
    }
    RETURN_IF_GED_ERROR(GED_SetSrc1Length(&instruction, sourceLength));
    RETURN_IF_GED_ERROR(GED_SetDescRegFile(&instruction, GED_REG_FILE_IMM));
    RETURN_IF_GED_ERROR(GED_SetMsgDesc(&instruction, value.desc));
    if (value.eot)
      RETURN_IF_GED_ERROR(GED_SetEOT(&instruction, GED_EOT_EOT));
    return setOptions(instruction, value.swsb, true, value.rawSwsb);
  }

  static GED_SYNC_FC getSyncFunction(SyncKind kind) {
    switch (kind) {
    case SyncKind::nop:
      return GED_SYNC_FC_nop;
    case SyncKind::allrd:
      return GED_SYNC_FC_allrd;
    case SyncKind::allwr:
      return GED_SYNC_FC_allwr;
    case SyncKind::bar:
      return GED_SYNC_FC_bar;
    }
    llvm_unreachable("unknown sync function");
  }

  LogicalResult encodeItem(const SyncInstruction &value,
                           ged_ins_t &instruction) {
    ExecutionInfo execution{1, 0, false};
    if (failed(initialize(instruction, GED_OPCODE_sync, execution)))
      return failure();
    RETURN_IF_GED_ERROR(
        GED_SetSyncFC(&instruction, getSyncFunction(value.kind)));
    RETURN_IF_GED_ERROR(GED_SetDstHorzStride(&instruction, 1));
    if (value.kind == SyncKind::bar || value.sbidMask != 0) {
      RETURN_IF_GED_ERROR(GED_SetSrc0RegFile(&instruction, GED_REG_FILE_IMM));
      RETURN_IF_GED_ERROR(GED_SetSrc0DataType(&instruction, GED_DATA_TYPE_ud));
      RETURN_IF_GED_ERROR(GED_SetImm(&instruction, value.sbidMask));
    } else {
      RETURN_IF_GED_ERROR(GED_SetSrc0RegFile(&instruction, GED_REG_FILE_ARF));
    }
    return setOptions(instruction, value.swsb, false);
  }

  FailureOr<int32_t> getBranchOffset(uint32_t label) {
    auto iterator = labelOffsets.find(label);
    if (iterator == labelOffsets.end()) {
      moduleOp.emitError() << "undefined emission label L" << label;
      return failure();
    }
    int64_t offset = static_cast<int64_t>(iterator->second) - currentPc;
    if (offset < std::numeric_limits<int32_t>::min() ||
        offset > std::numeric_limits<int32_t>::max()) {
      moduleOp.emitError("GED branch offset exceeds signed 32-bit range");
      return failure();
    }
    return static_cast<int32_t>(offset);
  }

  LogicalResult setBranchFields(ged_ins_t &instruction) {
    RETURN_IF_GED_ERROR(
        GED_SetBranchCtrl(&instruction, GED_BRANCH_CTRL_Normal));
    RETURN_IF_GED_ERROR(GED_SetDstRegFile(&instruction, GED_REG_FILE_ARF));
    RETURN_IF_GED_ERROR(GED_SetDstRegNum(&instruction, 0));
    RETURN_IF_GED_ERROR(GED_SetDstSubRegNum(&instruction, 0));
    return success();
  }

  LogicalResult encodeItem(const GotoInstruction &value,
                           ged_ins_t &instruction) {
    ExecutionInfo execution{32, 0, false};
    if (failed(initialize(instruction, GED_OPCODE_goto, execution,
                          value.predicate)) ||
        failed(setBranchFields(instruction)))
      return failure();
    if (!value.predicate) {
      RETURN_IF_GED_ERROR(GED_SetPredInv(&instruction, GED_PRED_INV_Normal));
      RETURN_IF_GED_ERROR(GED_SetFlagRegNum(&instruction, 0));
      RETURN_IF_GED_ERROR(GED_SetFlagSubRegNum(&instruction, 0));
    }
    RETURN_IF_GED_ERROR(GED_SetSrc0RegFile(&instruction, GED_REG_FILE_IMM));
    RETURN_IF_GED_ERROR(GED_SetSrc1RegFile(&instruction, GED_REG_FILE_IMM));
    FailureOr<int32_t> jip = getBranchOffset(value.jip);
    FailureOr<int32_t> uip = getBranchOffset(value.uip);
    if (failed(jip) || failed(uip))
      return failure();
    RETURN_IF_GED_ERROR(GED_SetJIP(&instruction, *jip));
    RETURN_IF_GED_ERROR(GED_SetUIP(&instruction, *uip));
    return setOptions(instruction, {}, false);
  }

  LogicalResult encodeItem(const JmpiInstruction &value,
                           ged_ins_t &instruction) {
    ExecutionInfo execution{1, 0, true};
    if (failed(initialize(instruction, GED_OPCODE_jmpi, execution,
                          value.predicate)))
      return failure();
    RETURN_IF_GED_ERROR(
        GED_SetBranchCtrl(&instruction, GED_BRANCH_CTRL_Normal));

    GED_REG_FILE ipFile;
    uint32_t ipNumber;
    uint32_t ipSubRegister;
    ArfReference ip{ARFFile::ip, 0, 0};
    if (failed(getRegister(ip, DataType::ud, ipFile, ipNumber, ipSubRegister)))
      return failure();
    RETURN_IF_GED_ERROR(GED_SetDstRegFile(&instruction, ipFile));
    RETURN_IF_GED_ERROR(GED_SetDstRegNum(&instruction, ipNumber));
    RETURN_IF_GED_ERROR(GED_SetDstSubRegNum(&instruction, ipSubRegister));
    RETURN_IF_GED_ERROR(GED_SetSrc0RegFile(&instruction, GED_REG_FILE_IMM));
    FailureOr<int32_t> target = getBranchOffset(value.target);
    if (failed(target))
      return failure();
    RETURN_IF_GED_ERROR(GED_SetJIP(&instruction, *target));
    return setOptions(instruction, {}, false);
  }

  LogicalResult encodeItem(const JoinInstruction &value,
                           ged_ins_t &instruction) {
    ExecutionInfo execution{32, 0, false};
    if (failed(initialize(instruction, GED_OPCODE_join, execution)) ||
        failed(setBranchFields(instruction)))
      return failure();
    RETURN_IF_GED_ERROR(GED_SetPredInv(&instruction, GED_PRED_INV_Normal));
    RETURN_IF_GED_ERROR(GED_SetFlagRegNum(&instruction, 0));
    RETURN_IF_GED_ERROR(GED_SetFlagSubRegNum(&instruction, 0));
    RETURN_IF_GED_ERROR(GED_SetSrc0RegFile(&instruction, GED_REG_FILE_IMM));
    FailureOr<int32_t> target = getBranchOffset(value.uip);
    if (failed(target))
      return failure();
    RETURN_IF_GED_ERROR(GED_SetJIP(&instruction, *target));
    return setOptions(instruction, {}, false);
  }

#undef RETURN_IF_GED_ERROR

  ModuleOp moduleOp;
  llvm::raw_ostream &output;
  GED_MODEL model;
  llvm::DenseMap<uint32_t, uint32_t> labelOffsets;
  uint32_t binarySize = 0;
  uint32_t currentPc = 0;
};

} // namespace
} // namespace inter::detail

namespace inter {

LogicalResult emitGedBinary(ModuleOp moduleOp, llvm::raw_ostream &output,
                            uint32_t *payloadEntryOffset) {
  func::FuncOp kernel;
  for (func::FuncOp function : moduleOp.getOps<func::FuncOp>())
    if (function->hasAttr(kTargetAttrName)) {
      kernel = function;
      break;
    }
  TargetAttr targetAttr =
      kernel ? kernel->getAttrOfType<TargetAttr>(kTargetAttrName)
             : TargetAttr{};
  llvm::Expected<TargetConfig> target = TargetConfig::resolve(targetAttr);
  if (!target)
    return moduleOp.emitError(llvm::toString(target.takeError())), failure();
  GED_MODEL model;
  switch (target->getArchitecture()) {
  case TargetArchitecture::xe2:
    model = GED_MODEL_XE2;
    break;
  }
  detail::EmissionProgram program;
  if (failed(detail::lowerToEmissionProgram(kernel, program)))
    return failure();
  detail::GedEncoder encoder(moduleOp, output, model);
  if (failed(encoder.encode(program)))
    return failure();
  if (payloadEntryOffset) {
    if (!program.payloadEntryLabel)
      return moduleOp.emitError("kernel has no payload prologue"), failure();
    std::optional<uint32_t> offset =
        encoder.getLabelOffset(*program.payloadEntryLabel);
    if (!offset)
      return moduleOp.emitError("payload entry label was not laid out"),
             failure();
    *payloadEntryOffset = *offset;
  }
  return success();
}

} // namespace inter
