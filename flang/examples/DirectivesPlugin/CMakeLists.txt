# TODO: Note that this is currently only available on Linux.
# On Windows, we would also have to specify e.g. `PLUGIN_TOOL`.
#
# Nothing is linked in on purpose: the registry the plugin adds its directives
# to is resolved against `flang -fc1` when the shared object is loaded.
add_llvm_example_library(flangDirectivesPlugin
    MODULE
    DirectivesPlugin.cpp
)
