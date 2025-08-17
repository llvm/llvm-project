include "llvm/Option/OptParser.td"

def LinkerOnlyOption : OptionFlag;

def help : Flag<["-", "--"], "help">,
  HelpText<"Display available options (--help-hidden for more)">;

def help_hidden : Flag<["-", "--"], "help-hidden">,
  HelpText<"Display all available options">;

def verbose : Flag<["-"], "v">, HelpText<"Print verbose information">;
def version : Flag<["--"], "version">,
  HelpText<"Display the version number and exit">;

def o : JoinedOrSeparate<["-"], "o">, MetaVarName<"<path>">,
  HelpText<"Path to file to write output">;
def output : Separate<["--"], "output-file">, Alias<o>, Flags<[HelpHidden]>,
  HelpText<"Alias for -o">;

def library_path_EQ : Joined<["--", "-"], "library-path=">,
  Flags<[HelpHidden]>, HelpText<"Add <dir> to the library search path">;

def device_libs_EQ : CommaJoined<["--", "-"], "device-libs=">,
  Flags<[LinkerOnlyOption]>,
  HelpText<"A comma separated list of device libraries that are linked during the device link.">;

def arch_EQ : Joined<["--", "-"], "arch=">,
              Flags<[LinkerOnlyOption]>,
              MetaVarName<"<arch>">,
              HelpText<"The device architecture">;
def triple_EQ : Joined<["--", "-"], "triple=">,
                Flags<[LinkerOnlyOption]>,
                MetaVarName<"<triple>">,
                HelpText<"The device target triple">;

def save_temps : Flag<["--", "-"], "save-temps">,
  Flags<[LinkerOnlyOption]>, HelpText<"Save intermediate results">;

def dry_run : Flag<["--", "-"], "dry-run">, Flags<[LinkerOnlyOption]>,
  HelpText<"Print generated commands without running.">;

def spirv_dump_device_code_EQ : Joined<["--", "-"], "spirv-dump-device-code=">,
  Flags<[LinkerOnlyOption]>,
  HelpText<"Path to the folder where the tool dumps SPIR-V device code. Other formats aren't dumped.">;

def print_linked_module : Flag<["--"], "print-linked-module">,
                          Flags<[LinkerOnlyOption]>,
                          HelpText<"Print the linked module's IR for testing">;

def ocloc_options_EQ : Joined<["--", "-"], "ocloc-options=">,
  Flags<[LinkerOnlyOption]>,
  HelpText<"Options passed to ocloc for Intel GPU AOT compilation">;

def opencl_aot_options_EQ : Joined<["--", "-"], "opencl-aot-options=">,
  Flags<[LinkerOnlyOption]>,
  HelpText<"Options passed to opencl-aot for Intel CPU AOT compilation">;
