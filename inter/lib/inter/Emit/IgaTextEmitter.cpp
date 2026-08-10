#include "EmissionProgram.h"
#include "inter/Emit/Emit.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <variant>

using namespace inter::xemachine;

namespace inter::detail {
namespace {

class IgaTextPrinter {
public:
  explicit IgaTextPrinter(llvm::raw_ostream &output) : output(output) {}

  void print(const EmissionProgram &program) {
    for (const EmissionItem &item : program.items)
      std::visit([&](const auto &value) { printItem(value); }, item);
  }

private:
  static llvm::StringRef getTypeSuffix(DataType type) {
    switch (type) {
    case DataType::ub:
      return "ub";
    case DataType::uw:
      return "uw";
    case DataType::ud:
      return "ud";
    case DataType::q:
      return "q";
    case DataType::f:
      return "f";
    }
    llvm_unreachable("unknown data type");
  }

  static llvm::StringRef getOpcodeName(AluOpcode opcode) {
    switch (opcode) {
    case AluOpcode::mov:
      return "mov";
    case AluOpcode::add:
      return "add";
    case AluOpcode::shl:
      return "shl";
    case AluOpcode::and_:
      return "and";
    case AluOpcode::or_:
      return "or";
    case AluOpcode::add3:
      return "add3";
    case AluOpcode::mul:
      return "mul";
    }
    llvm_unreachable("unknown ALU opcode");
  }

  void printGrf(GrfReference reference) {
    output << "r" << reference.number << "." << reference.sub;
  }

  void printArf(ArfReference reference) {
    output << stringifyARFFile(reference.file) << reference.number << "."
           << reference.sub;
  }

  void printRegister(const RegisterReference &reference) {
    std::visit([&](const auto &value) { printRegisterValue(value); },
               reference);
  }

  void printRegisterValue(GrfReference reference) { printGrf(reference); }

  void printRegisterValue(ArfReference reference) { printArf(reference); }

  void printSource(const SourceOperand &source) {
    if (source.negate)
      output << "-";
    if (const auto *immediate = std::get_if<Immediate>(&source.value)) {
      output << "0x" << llvm::utohexstr(immediate->value, true) << ":"
             << getTypeSuffix(immediate->type);
      return;
    }
    if (const auto *grf = std::get_if<GrfReference>(&source.value))
      printGrf(*grf);
    else
      printArf(std::get<ArfReference>(source.value));
    output << "<" << source.region.vstride << ";" << source.region.width << ","
           << source.region.hstride << ">:" << getTypeSuffix(source.type);
  }

  void printSwsb(const SwsbInfo &swsb, bool eot = false) {
    if (!eot && swsb.distance < 0 && swsb.token < 0)
      return;

    output << "{";
    bool needsComma = false;
    if (eot) {
      output << "EOT";
      needsComma = true;
    }
    if (swsb.distance >= 0) {
      if (needsComma)
        output << ",";
      output << (swsb.pipe == DistancePipe::all ? "A@" : "I@") << swsb.distance;
      needsComma = true;
    }
    if (swsb.token >= 0) {
      if (needsComma)
        output << ",";
      output << "$" << swsb.token;
    }
    output << "}";
  }

  void printItem(const Label &label) { output << "L" << label.id << ":\n"; }

  void printItem(const AluInstruction &instruction) {
    output << (instruction.execution.noMask ? "(W)     " : "        ");
    output << getOpcodeName(instruction.opcode) << " ("
           << instruction.execution.size << "|M"
           << instruction.execution.maskOffset << ")  ";
    if (instruction.destination) {
      printRegister(instruction.destination->value);
      output << "<" << instruction.destination->hstride << ">";
    } else {
      output << "null<1>";
    }
    output << ":" << getTypeSuffix(instruction.destinationType) << "  ";
    for (const SourceOperand &source : instruction.sources) {
      printSource(source);
      output << "  ";
    }
    printSwsb(instruction.swsb);
    output << "\n";
  }

  void printItem(const CompareInstruction &instruction) {
    output << "        cmp (" << instruction.execution.size << "|M"
           << instruction.execution.maskOffset << ")  ("
           << stringifyCondModifier(instruction.condition) << ")";
    printArf(instruction.flag);
    output << "   null<1>:" << getTypeSuffix(instruction.dataType) << "  ";
    for (const SourceOperand &source : instruction.sources) {
      printSource(source);
      output << "  ";
    }
    printSwsb(instruction.swsb);
    output << "\n";
  }

  void printItem(const SendInstruction &instruction) {
    output << (instruction.execution.noMask ? "(W)     " : "        ");
    output << "send." << stringifySendFn(instruction.function) << " ("
           << instruction.execution.size << "|M"
           << instruction.execution.maskOffset << ")  ";
    if (instruction.destination)
      printGrf(*instruction.destination);
    else
      output << "null";
    output << "  ";
    printGrf(instruction.address);
    output << "  ";
    if (instruction.data) {
      printGrf(*instruction.data);
      if (instruction.data->widthDwords > 16)
        output << ":" << instruction.data->widthDwords / 16;
    } else {
      output << "null:0";
    }
    output << "  0x" << llvm::utohexstr(instruction.exdesc, true) << "  0x"
           << llvm::utohexstr(instruction.desc, true) << "           ";
    printSwsb(instruction.swsb, instruction.eot);
    output << "\n";
  }

  void printItem(const SyncInstruction &instruction) {
    output << "        sync." << stringifySyncKind(instruction.kind);
    if (instruction.kind == SyncKind::bar)
      output << " 0x0\n";
    else
      output << " null\n";
  }

  void printItem(const GotoInstruction &instruction) {
    output << "        ";
    if (instruction.predicate) {
      output << "(";
      if (instruction.predicate->inverse)
        output << "~";
      printArf(instruction.predicate->flag);
      output << ") ";
    }
    output << "goto (32|M0)  L" << instruction.jip << "  L" << instruction.uip
           << "\n";
  }

  void printItem(const JoinInstruction &instruction) {
    output << "        join (32|M0)  L" << instruction.uip << "\n";
  }

  llvm::raw_ostream &output;
};

} // namespace

void printIgaAsm(const EmissionProgram &program, llvm::raw_ostream &output) {
  IgaTextPrinter(output).print(program);
}

} // namespace inter::detail

namespace inter {

mlir::LogicalResult emitIgaAsm(mlir::ModuleOp moduleOp,
                               llvm::raw_ostream &output) {
  detail::EmissionProgram program;
  if (failed(detail::lowerToEmissionProgram(moduleOp, program)))
    return mlir::failure();
  detail::printIgaAsm(program, output);
  return mlir::success();
}

} // namespace inter
