add_llvm_example_library(flangFeatureList
    MODULE
    FeatureList.cpp

    DEPENDS
    acc_gen
    flangFrontend
    omp_gen
)
