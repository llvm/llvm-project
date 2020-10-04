// ---------------- llvm specific headers ---------------- //
#include "llvm/ADT/APInt.h"
#include "llvm/IR/Verifier.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

// --------------------- CPP headers ----------------------- //
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace llvm;

static Function *CreateFunctions(Module *M, LLVMContext &Context) {
	//the function definition
  FunctionType *FibTy = FunctionType::get(Type::getInt32Ty(Context),
                                           {Type::getInt32Ty(Context)}, false);
                                           
       //cast the function FibTy to module M
  Function *FibF =
      Function::Create(FibTy, Function::ExternalLinkage, "fib", M);

	//create a entry basic block 
  BasicBlock *BB = BasicBlock::Create(Context, "EntryBlock", FibF);

	//defining two constant integer variables
  	Value *One = ConstantInt::get(Type::getInt32Ty(Context), 1);
  	Value *Two = ConstantInt::get(Type::getInt32Ty(Context), 2);

//taking up the argument and setting its name
  Argument *ArgX = &*FibF->arg_begin();
  ArgX->setName("x");

//creating the return block
  BasicBlock *RetBB1 = BasicBlock::Create(Context, "return_one", FibF);

//creating the recursive block
  BasicBlock* RecurseBB = BasicBlock::Create(Context, "recurse", FibF);


  
  //checking if the given input is less than or equal to two 
  Value *CondInst1 = new ICmpInst(*BB, ICmpInst::ICMP_SLE, ArgX, Two, "cond1");
  //if the input is Two then return one(since first and second fibonacci number is 1)
  BranchInst::Create(RetBB1, RecurseBB, CondInst1, BB);
  ReturnInst::Create(Context, One, RetBB1);

//otherwise return fibonacci(arg-1)+fibonacci(arg-2)
//fibonacci(n-1) 
  Value *Sub1 = BinaryOperator::CreateSub(ArgX, One, "arg", RecurseBB);
  CallInst *CallFib1 = CallInst::Create(FibF, Sub1, "fib1", RecurseBB);
  CallFib1->setTailCall();
  
  //fibonacci(n-2)
 Value *Sub2 = BinaryOperator::CreateSub(ArgX, Two, "arg", RecurseBB);
  CallInst *CallFib2 = CallInst::Create(FibF, Sub2, "fib2", RecurseBB);
  CallFib2->setTailCall();

  Value *Ret = BinaryOperator::CreateAdd(CallFib1, CallFib2, "ret", RecurseBB);

  ReturnInst::Create(Context, Ret, RecurseBB);

  return FibF;
}

int main(int argc, char **argv) {
//trying to find out 4th fibonacci number
  int n = argc > 1 ? atol(argv[1]) : 4;

  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  LLVMContext Context;

 //creating the Module
  std::unique_ptr<Module> Owner(new Module("test", Context));
  Module *M = Owner.get();
	
//creating the fibonacci function with the module
   Function *FibF=CreateFunctions(M, Context);
	
std::string errStr;
  ExecutionEngine *EE =
    EngineBuilder(std::move(Owner))
    .setErrorStr(&errStr)
    .create();

  if (!EE) {
    errs() << argv[0] << ": We Failed to construct ExecutionEngine: " << errStr
           << "\n";
    return 1;
  }	
//Verifying the module
  errs() << "verifying the module ";
  if (verifyModule(*M)) {
    errs() << argv[0] << ": Error occured in constructing function!\n";
    return 1;
  }

  errs() << "All OK\n";
  errs() << "We have constructed this LLVM module:\n\n---------\n" << *M;
  
   // Call the Fibonacci function with argument n:
  std::vector<GenericValue> Args(1);
  Args[0].IntVal = APInt(32, n);
  GenericValue GV = EE->runFunction(FibF, Args);

  // import result of execution
  outs() << "The "<< n<<"th fibonacci number is : " << GV.IntVal << "\n";
  return 0;
}
