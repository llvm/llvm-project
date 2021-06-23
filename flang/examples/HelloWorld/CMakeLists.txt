add_llvm_library(
    flangHelloWorldPlugin
    MODULE
    HelloWorldPlugin.cpp

    DEPENDS
    clangBasic

    LINK_COMPONENTS
    Option
    Support
)
