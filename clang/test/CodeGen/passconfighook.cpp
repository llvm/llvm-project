// RUN: %clangxx -shared -fPIC -I??/install/include -L%llvmshlibdir %s -o %t.so
// RUN: %clangxx -O3 -DMAIN -Xclang -load -Xclang %t.so %s -o %t-main | FileCheck %s

#ifndef MAIN

#include <llvm/Target/TargetMachine.h>
#include <llvm/CodeGen/TargetPassConfig.h>
#include <llvm/CodeGen/MachineFunctionPass.h>
#include <llvm/CodeGen/Passes.h>

#define DEBUG_TYPE "codegen-test"
#define CODEGEN_TEST_NAME "CodeGen Test Pass"

using namespace llvm;

namespace llvm {
    void initializeCodeGenTestPass(PassRegistry &);
} // namespace llvm

class CodeGenTest : public MachineFunctionPass {
public:
    static char ID;

    CodeGenTest(): MachineFunctionPass(ID) {
    }

    bool runOnMachineFunction(MachineFunction &MF) override {
        outs() << "[CodeGen] CodeGenTest::runOnMachineFunction" << "\n";
        return true;
    }

    StringRef getPassName() const override {
        return CODEGEN_TEST_NAME;
    }
};

char CodeGenTest::ID = 0;
INITIALIZE_PASS(CodeGenTest, DEBUG_TYPE, CODEGEN_TEST_NAME, false, false)

__attribute__((constructor)) static void initCodeGenPlugin() {
    initializeCodeGenTestPass(*PassRegistry::getPassRegistry());

    TargetMachine::registerTargetPassConfigCallback([](auto &TM, auto &PM, auto *TPC) {
        outs() << "registerTargetPassConfigCallback\n";
        TPC->insertPass(&GCLoweringID, &CodeGenTest::ID);
    });
}

#else

// CHECK: CodeGenTest::runOnMachineFunction
int main(int argc, char **argv) {
    return 0;
}

#endif
