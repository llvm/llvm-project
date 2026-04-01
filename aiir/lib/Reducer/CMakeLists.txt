add_aiir_library(AIIRReduce
   OptReductionPass.cpp
   ReductionNode.cpp
   ReductionTreePass.cpp
   Tester.cpp

   LINK_LIBS PUBLIC
   AIIRIR
   AIIRPass
   AIIRRewrite
   AIIRTransformUtils

   DEPENDS
   AIIRReducerIncGen
   AIIRDialectReductionPatternInterfaceIncGen
)

aiir_check_all_link_libraries(AIIRReduce)
