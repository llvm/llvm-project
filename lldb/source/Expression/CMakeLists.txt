set(LLVM_NO_RTTI 1)

add_lldb_library(lldbExpression
  ASTDumper.cpp
  ASTResultSynthesizer.cpp
  ASTStructExtractor.cpp
  ClangASTSource.cpp
  ClangExpressionDeclMap.cpp
  ClangExpressionParser.cpp
  ClangExpressionVariable.cpp
  ClangFunction.cpp
  ClangPersistentVariables.cpp
  ClangUserExpression.cpp
  ClangUtilityFunction.cpp
  DWARFExpression.cpp
  ExpressionSourceCode.cpp
  IRDynamicChecks.cpp
  IRExecutionUnit.cpp
  IRForTarget.cpp
  IRInterpreter.cpp
  IRMemoryMap.cpp
  Materializer.cpp
  )
