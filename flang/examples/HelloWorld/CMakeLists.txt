# TODO: Note that this is currently only available on Linux.
# On Windows, we would also have to specify e.g. `PLUGIN_TOOL`.
add_llvm_library(
    flangHelloWorldPlugin
    MODULE
    HelloWorldPlugin.cpp
)
