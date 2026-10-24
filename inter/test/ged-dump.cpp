#include "ged.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

static const char *getOpcodeName(GED_OPCODE opcode) {
  switch (opcode) {
  case GED_OPCODE_mov:
    return "mov";
  case GED_OPCODE_add:
    return "add";
  case GED_OPCODE_shl:
    return "shl";
  case GED_OPCODE_shr:
    return "shr";
  case GED_OPCODE_and:
    return "and";
  case GED_OPCODE_or:
    return "or";
  case GED_OPCODE_add3:
    return "add3";
  case GED_OPCODE_csel:
    return "csel";
  case GED_OPCODE_mul:
    return "mul";
  case GED_OPCODE_dpas:
    return "dpas";
  case GED_OPCODE_cmp:
    return "cmp";
  case GED_OPCODE_send:
    return "send";
  case GED_OPCODE_sync:
    return "sync";
  case GED_OPCODE_goto:
    return "goto";
  case GED_OPCODE_jmpi:
    return "jmpi";
  case GED_OPCODE_join:
    return "join";
  default:
    return "unknown";
  }
}

static const char *getSfidName(GED_SFID sfid) {
  switch (sfid) {
  case GED_SFID_UGM:
    return "ugm";
  case GED_SFID_GATEWAY:
    return "gateway";
  case GED_SFID_TGM:
    return "tgm";
  case GED_SFID_SLM:
    return "slm";
  default:
    return "unknown";
  }
}

static const char *getSyncName(GED_SYNC_FC function) {
  switch (function) {
  case GED_SYNC_FC_nop:
    return "nop";
  case GED_SYNC_FC_allrd:
    return "allrd";
  case GED_SYNC_FC_allwr:
    return "allwr";
  case GED_SYNC_FC_bar:
    return "bar";
  default:
    return "unknown";
  }
}

static const char *getRegFileName(GED_REG_FILE file) {
  switch (file) {
  case GED_REG_FILE_ARF:
    return "arf";
  case GED_REG_FILE_GRF:
    return "grf";
  case GED_REG_FILE_IMM:
    return "imm";
  default:
    return "unknown";
  }
}

static const char *getDataTypeName(GED_DATA_TYPE type) {
  switch (type) {
  case GED_DATA_TYPE_ub:
    return "ub";
  case GED_DATA_TYPE_b:
    return "b";
  case GED_DATA_TYPE_uw:
    return "uw";
  case GED_DATA_TYPE_w:
    return "w";
  case GED_DATA_TYPE_ud:
    return "ud";
  case GED_DATA_TYPE_d:
    return "d";
  case GED_DATA_TYPE_q:
    return "q";
  case GED_DATA_TYPE_f:
    return "f";
  default:
    return "unknown";
  }
}

static const char *getPrecisionName(GED_PRECISION precision) {
  switch (precision) {
  case GED_PRECISION_f16:
    return "f16";
  case GED_PRECISION_bf16:
    return "bf16";
  default:
    return "unknown";
  }
}

static const char *getMaskName(GED_MASK_CTRL mask) {
  return mask == GED_MASK_CTRL_NoMask ? "nomask" : "normal";
}

static const char *getConditionName(GED_COND_MODIFIER condition) {
  switch (condition) {
  case GED_COND_MODIFIER_z:
    return "eq";
  case GED_COND_MODIFIER_nz:
    return "ne";
  case GED_COND_MODIFIER_l:
    return "lt";
  case GED_COND_MODIFIER_le:
    return "le";
  case GED_COND_MODIFIER_g:
    return "gt";
  case GED_COND_MODIFIER_ge:
    return "ge";
  default:
    return "normal";
  }
}

static uint32_t getChannelOffset(GED_CHANNEL_OFFSET offset) {
  switch (offset) {
  case GED_CHANNEL_OFFSET_M0:
    return 0;
  case GED_CHANNEL_OFFSET_M8:
    return 8;
  case GED_CHANNEL_OFFSET_M16:
    return 16;
  case GED_CHANNEL_OFFSET_M24:
    return 24;
  default:
    return ~0U;
  }
}

template <typename T>
static T getField(T (*getter)(ged_ins_t *, GED_RETURN_VALUE *),
                  ged_ins_t &instruction, const char *name) {
  GED_RETURN_VALUE result;
  T value = getter(&instruction, &result);
  if (result != GED_RETURN_VALUE_SUCCESS) {
    std::cerr << name << ": " << GED_GetReturnValueString(result) << "\n";
    std::exit(1);
  }
  return value;
}

static void printSource(ged_ins_t &instruction, uint32_t index, bool ternary) {
  GED_REG_FILE file;
  GED_DATA_TYPE type;
  uint32_t number = 0;
  uint32_t sub = 0;
  uint32_t vertical = 0;
  uint32_t width = 0;
  uint32_t horizontal = 0;
  GED_SRC_MOD modifier = GED_SRC_MOD_Normal;
  if (index == 0) {
    file =
        getField<GED_REG_FILE>(GED_GetSrc0RegFile, instruction, "Src0RegFile");
    type = getField<GED_DATA_TYPE>(GED_GetSrc0DataType, instruction,
                                   "Src0DataType");
    if (file != GED_REG_FILE_IMM) {
      number = getField<uint32_t>(GED_GetSrc0RegNum, instruction, "Src0RegNum");
      sub = getField<uint32_t>(GED_GetSrc0SubRegNum, instruction,
                               "Src0SubRegNum");
      modifier =
          getField<GED_SRC_MOD>(GED_GetSrc0SrcMod, instruction, "Src0SrcMod");
      vertical = getField<uint32_t>(GED_GetSrc0VertStride, instruction,
                                    "Src0VertStride");
      horizontal = getField<uint32_t>(GED_GetSrc0HorzStride, instruction,
                                      "Src0HorzStride");
      if (!ternary)
        width = getField<uint32_t>(GED_GetSrc0Width, instruction, "Src0Width");
    }
  } else if (index == 1) {
    file =
        getField<GED_REG_FILE>(GED_GetSrc1RegFile, instruction, "Src1RegFile");
    type = getField<GED_DATA_TYPE>(GED_GetSrc1DataType, instruction,
                                   "Src1DataType");
    if (file != GED_REG_FILE_IMM) {
      number = getField<uint32_t>(GED_GetSrc1RegNum, instruction, "Src1RegNum");
      sub = getField<uint32_t>(GED_GetSrc1SubRegNum, instruction,
                               "Src1SubRegNum");
      modifier =
          getField<GED_SRC_MOD>(GED_GetSrc1SrcMod, instruction, "Src1SrcMod");
      vertical = getField<uint32_t>(GED_GetSrc1VertStride, instruction,
                                    "Src1VertStride");
      horizontal = getField<uint32_t>(GED_GetSrc1HorzStride, instruction,
                                      "Src1HorzStride");
      if (!ternary)
        width = getField<uint32_t>(GED_GetSrc1Width, instruction, "Src1Width");
    }
  } else {
    file =
        getField<GED_REG_FILE>(GED_GetSrc2RegFile, instruction, "Src2RegFile");
    type = getField<GED_DATA_TYPE>(GED_GetSrc2DataType, instruction,
                                   "Src2DataType");
    if (file != GED_REG_FILE_IMM) {
      number = getField<uint32_t>(GED_GetSrc2RegNum, instruction, "Src2RegNum");
      sub = getField<uint32_t>(GED_GetSrc2SubRegNum, instruction,
                               "Src2SubRegNum");
      modifier =
          getField<GED_SRC_MOD>(GED_GetSrc2SrcMod, instruction, "Src2SrcMod");
      horizontal = getField<uint32_t>(GED_GetSrc2HorzStride, instruction,
                                      "Src2HorzStride");
    }
  }

  std::cout << " src" << index << "=" << getRegFileName(file);
  if (file == GED_REG_FILE_IMM) {
    uint64_t immediate =
        ternary && index == 0
            ? getField<uint64_t>(GED_GetSrc0TernaryImm, instruction,
                                 "Src0TernaryImm")
        : ternary && index == 2
            ? getField<uint64_t>(GED_GetSrc2TernaryImm, instruction,
                                 "Src2TernaryImm")
            : getField<uint64_t>(GED_GetImm, instruction, "Imm");
    std::cout << "0x" << std::hex << immediate;
  } else {
    std::cout << number << "." << sub;
  }
  std::cout << ":" << getDataTypeName(type);
  if (modifier == GED_SRC_MOD_Negative)
    std::cout << ":neg";
  if (file != GED_REG_FILE_IMM) {
    std::cout << "<" << vertical;
    if (!ternary)
      std::cout << ";" << width << ",";
    else
      std::cout << ";";
    std::cout << horizontal << ">";
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: inter-ged-dump <binary>\n";
    return 1;
  }

  std::ifstream input(argv[1], std::ios::binary);
  std::vector<unsigned char> binary((std::istreambuf_iterator<char>(input)),
                                    std::istreambuf_iterator<char>());
  if (input.bad() || binary.empty() || binary.size() % 16 != 0) {
    std::cerr << "input must contain native 16-byte GED instructions\n";
    return 1;
  }

  for (size_t pc = 0; pc < binary.size(); pc += 16) {
    ged_ins_t instruction;
    GED_RETURN_VALUE result =
        GED_DecodeIns(GED_MODEL_XE2, binary.data() + pc, 16, &instruction);
    if (result != GED_RETURN_VALUE_SUCCESS || GED_InsSize(&instruction) != 16) {
      std::cerr << "decode failed at byte " << pc << ": "
                << GED_GetReturnValueString(result) << "\n";
      return 1;
    }

    GED_OPCODE opcode = GED_GetOpcode(&instruction);
    uint32_t executionSize =
        getField<uint32_t>(GED_GetExecSize, instruction, "ExecSize");
    uint32_t swsb = getField<uint32_t>(GED_GetSWSB, instruction, "SWSB");
    std::cout << "pc=" << pc << " opcode=" << getOpcodeName(opcode)
              << " exec=" << executionSize << " swsb=0x" << std::hex << swsb;
    if (opcode == GED_OPCODE_send) {
      GED_SFID sfid = getField<GED_SFID>(GED_GetSFID, instruction, "SFID");
      uint32_t desc =
          getField<uint32_t>(GED_GetMsgDesc, instruction, "MsgDesc");
      GED_REG_FILE exdescFile = getField<GED_REG_FILE>(
          GED_GetExDescRegFile, instruction, "ExDescRegFile");
      std::cout << " sfid=" << getSfidName(sfid)
                << " exdescRegFile=" << getRegFileName(exdescFile);
      if (exdescFile == GED_REG_FILE_IMM) {
        uint32_t exdesc = getField<uint32_t>(GED_GetExMsgDescImm, instruction,
                                             "ExMsgDescImm");
        std::cout << " exdesc=0x" << exdesc;
      } else {
        uint32_t sub = getField<uint32_t>(GED_GetExDescAddrSubRegNum,
                                          instruction, "ExDescAddrSubRegNum");
        std::cout << " exdescAddrSubRegNum=" << sub / 2
                  << " exdescAddrSubRegRaw=" << sub;
      }
      std::cout << " desc=0x" << desc;
    } else if (opcode == GED_OPCODE_sync) {
      GED_SYNC_FC function =
          getField<GED_SYNC_FC>(GED_GetSyncFC, instruction, "SyncFC");
      std::cout << " function=" << getSyncName(function);
    } else if (opcode == GED_OPCODE_goto) {
      int32_t jip = getField<int32_t>(GED_GetJIP, instruction, "JIP");
      int32_t uip = getField<int32_t>(GED_GetUIP, instruction, "UIP");
      std::cout << std::dec << " jip=" << jip << " uip=" << uip;
    } else if (opcode == GED_OPCODE_jmpi) {
      int32_t jip = getField<int32_t>(GED_GetJIP, instruction, "JIP");
      std::cout << std::dec << " jip=" << jip;
    } else if (opcode == GED_OPCODE_join) {
      int32_t jip = getField<int32_t>(GED_GetJIP, instruction, "JIP");
      std::cout << std::dec << " jip=" << jip;
    } else if (opcode == GED_OPCODE_dpas) {
      uint32_t depth = getField<uint32_t>(GED_GetSystolicDepth, instruction,
                                          "SystolicDepth");
      uint32_t repeat =
          getField<uint32_t>(GED_GetRepeatCount, instruction, "RepeatCount");
      GED_PRECISION b = getField<GED_PRECISION>(GED_GetSrc1Precision,
                                                instruction, "Src1Precision");
      GED_PRECISION a = getField<GED_PRECISION>(GED_GetSrc2Precision,
                                                instruction, "Src2Precision");
      std::cout << std::dec << " depth=" << depth << " repeat=" << repeat
                << " bPrecision=" << getPrecisionName(b)
                << " aPrecision=" << getPrecisionName(a);
    }
    std::cout << std::dec;
    GED_MASK_CTRL mask =
        getField<GED_MASK_CTRL>(GED_GetMaskCtrl, instruction, "MaskCtrl");
    GED_CHANNEL_OFFSET channel = getField<GED_CHANNEL_OFFSET>(
        GED_GetChannelOffset, instruction, "ChannelOffset");
    GED_PRED_CTRL predicate =
        getField<GED_PRED_CTRL>(GED_GetPredCtrl, instruction, "PredCtrl");
    std::cout << " mask=" << getMaskName(mask)
              << " channel=" << getChannelOffset(channel) << " pred="
              << (predicate == GED_PRED_CTRL_Sequential ? "sequential"
                                                        : "normal");
    if (predicate == GED_PRED_CTRL_Sequential) {
      GED_PRED_INV inverse =
          getField<GED_PRED_INV>(GED_GetPredInv, instruction, "PredInv");
      uint32_t flag =
          getField<uint32_t>(GED_GetFlagRegNum, instruction, "FlagRegNum");
      uint32_t sub = getField<uint32_t>(GED_GetFlagSubRegNum, instruction,
                                        "FlagSubRegNum");
      std::cout << " inverse=" << (inverse == GED_PRED_INV_Invert)
                << " flag=" << flag << "." << sub;
    }
    if (opcode == GED_OPCODE_cmp || opcode == GED_OPCODE_csel) {
      GED_COND_MODIFIER condition = getField<GED_COND_MODIFIER>(
          GED_GetCondModifier, instruction, "CondModifier");
      uint32_t flag =
          getField<uint32_t>(GED_GetFlagRegNum, instruction, "FlagRegNum");
      uint32_t sub = getField<uint32_t>(GED_GetFlagSubRegNum, instruction,
                                        "FlagSubRegNum");
      std::cout << " condition=" << getConditionName(condition)
                << " flag=" << flag << "." << sub;
    }
    if (opcode == GED_OPCODE_send) {
      GED_REG_FILE destinationFile =
          getField<GED_REG_FILE>(GED_GetDstRegFile, instruction, "DstRegFile");
      uint32_t destination =
          getField<uint32_t>(GED_GetDstRegNum, instruction, "DstRegNum");
      uint32_t source0 =
          getField<uint32_t>(GED_GetSrc0RegNum, instruction, "Src0RegNum");
      GED_REG_FILE source1File = getField<GED_REG_FILE>(
          GED_GetSrc1RegFile, instruction, "Src1RegFile");
      uint32_t source1 =
          source1File == GED_REG_FILE_GRF
              ? getField<uint32_t>(GED_GetSrc1RegNum, instruction, "Src1RegNum")
              : 0;
      uint32_t sourceLength =
          getField<uint32_t>(GED_GetSrc1Length, instruction, "Src1Length");
      GED_EOT eot = getField<GED_EOT>(GED_GetEOT, instruction, "EOT");
      std::cout << " dst=" << getRegFileName(destinationFile) << destination
                << " src0=grf" << source0
                << " src1=" << getRegFileName(source1File) << source1
                << " len=" << sourceLength << " eot=" << (eot == GED_EOT_EOT);
    } else if (opcode == GED_OPCODE_dpas) {
      uint32_t destination =
          getField<uint32_t>(GED_GetDstRegNum, instruction, "DstRegNum");
      uint32_t acc =
          getField<uint32_t>(GED_GetSrc0RegNum, instruction, "Src0RegNum");
      uint32_t b =
          getField<uint32_t>(GED_GetSrc1RegNum, instruction, "Src1RegNum");
      uint32_t a =
          getField<uint32_t>(GED_GetSrc2RegNum, instruction, "Src2RegNum");
      std::cout << " dst=grf" << destination << " acc=grf" << acc << " b=grf"
                << b << " a=grf" << a;
    } else if (opcode == GED_OPCODE_mov || opcode == GED_OPCODE_add ||
               opcode == GED_OPCODE_shl || opcode == GED_OPCODE_shr ||
               opcode == GED_OPCODE_and || opcode == GED_OPCODE_or ||
               opcode == GED_OPCODE_add3 || opcode == GED_OPCODE_csel ||
               opcode == GED_OPCODE_mul || opcode == GED_OPCODE_cmp) {
      GED_REG_FILE destinationFile =
          getField<GED_REG_FILE>(GED_GetDstRegFile, instruction, "DstRegFile");
      GED_DATA_TYPE destinationType = getField<GED_DATA_TYPE>(
          GED_GetDstDataType, instruction, "DstDataType");
      uint32_t destination =
          getField<uint32_t>(GED_GetDstRegNum, instruction, "DstRegNum");
      uint32_t destinationSub =
          getField<uint32_t>(GED_GetDstSubRegNum, instruction, "DstSubRegNum");
      uint32_t destinationStride = getField<uint32_t>(
          GED_GetDstHorzStride, instruction, "DstHorzStride");
      std::cout << " dst=" << getRegFileName(destinationFile) << destination
                << "." << destinationSub << ":"
                << getDataTypeName(destinationType) << "<" << destinationStride
                << ">";
      bool ternary = opcode == GED_OPCODE_add3 || opcode == GED_OPCODE_csel;
      printSource(instruction, 0, ternary);
      if (opcode != GED_OPCODE_mov)
        printSource(instruction, 1, ternary);
      if (ternary)
        printSource(instruction, 2, true);
    }
    std::cout << std::dec << "\n";
  }
  return 0;
}
