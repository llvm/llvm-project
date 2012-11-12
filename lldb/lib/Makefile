##===- source/Makefile -------------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL := ../../..
LLDB_LEVEL := ..

LIBRARYNAME = lldb

#EXPORTED_SYMBOL_FILE = $(PROJ_SRC_DIR)/$(LLDB_LEVEL)/resources/lldb-framework-exports
NO_BUILD_ARCHIVE = 1
LINK_LIBS_IN_SHARED = 1
SHARED_LIBRARY = 1

PYTHON_BUILD_FLAGS = $(shell python-config --ldflags)

# Include all archives in liblldb.so file
USEDLIBS = lldbAPI.a \
	lldbBreakpoint.a \
	lldbCommands.a \
	lldbCore.a \
	lldbExpression.a \
	lldbHostCommon.a \
	lldbInitAndLog.a \
	lldbInterpreter.a \
	lldbPluginABIMacOSX_arm.a \
	lldbPluginABIMacOSX_i386.a \
	lldbPluginABISysV_x86_64.a \
	lldbPluginDisassemblerLLVM.a \
	lldbPluginDynamicLoaderStatic.a \
	lldbPluginDynamicLoaderPOSIX.a \
	lldbPluginEmulateInstructionARM.a \
	lldbPluginLanguageRuntimeCPlusPlusItaniumABI.a \
	lldbPluginLanguageRuntimeObjCAppleObjCRuntime.a \
	lldbPluginObjectContainerBSDArchive.a \
	lldbPluginObjectFileELF.a \
	lldbPluginObjectFilePECOFF.a \
	lldbPluginOperatingSystemPython.a \
	lldbPluginPlatformGDBServer.a \
	lldbPluginProcessGDBRemote.a \
	lldbPluginSymbolFileDWARF.a \
	lldbPluginSymbolFileSymtab.a \
	lldbPluginUnwindAssemblyInstEmulation.a \
	lldbPluginUnwindAssemblyx86.a \
	lldbPluginUtility.a \
	lldbSymbol.a \
	lldbTarget.a \
	lldbUtility.a \
	clangAnalysis.a \
	clangAST.a \
	clangBasic.a \
	clangCodeGen.a \
	clangFrontend.a \
	clangDriver.a \
	clangEdit.a \
	clangLex.a \
	clangParse.a \
	clangSema.a \
	clangSerialization.a \
	LLVMMCDisassembler.a \
	lldbPluginPlatformMacOSX.a \
	lldbPluginPlatformLinux.a \
	lldbPluginPlatformFreeBSD.a

include $(LLDB_LEVEL)/../../Makefile.config

LINK_COMPONENTS := $(TARGETS_TO_BUILD) asmparser bitreader bitwriter codegen \
                   instrumentation ipo selectiondag jit mc mcjit linker

ifeq ($(HOST_OS),Darwin)
  USEDLIBS += lldbHostMacOSX.a \
              lldbPluginDynamicLoaderMacOSX.a \
              lldbPluginDynamicLoaderDarwinKernel.a \
              lldbPluginObjectContainerUniversalMachO.a \
              lldbPluginObjectFileMachO.a \
              lldbPluginOperatingSystemDarwinKernel.a \
              lldbPluginSymbolVendorMacOSX.a \
              lldbPluginProcessDarwin.a \
              lldbPluginProcessMachCore.a
endif

ifeq ($(HOST_OS),Linux)
  USEDLIBS += lldbHostLinux.a \
              lldbPluginProcessLinux.a \
              lldbPluginProcessPOSIX.a \
              lldbPluginDynamicLoaderMacOSX.a \
              lldbPluginDynamicLoaderDarwinKernel.a \
              lldbPluginOperatingSystemDarwinKernel.a

endif

ifeq ($(HOST_OS),FreeBSD)
  USEDLIBS += lldbHostFreeBSD.a \
              lldbPluginProcessPOSIX.a \
              lldbPluginProcessFreeBSD.a
endif

include $(LEVEL)/Makefile.common

ifeq ($(HOST_OS),Darwin)
    LLVMLibsOptions += -Wl,-all_load
    # set dylib internal version number to llvmCore submission number
    ifdef LLDB_SUBMIT_VERSION
        LLVMLibsOptions += -Wl,-current_version \
                           -Wl,$(LLDB_SUBMIT_VERSION).$(LLDB_SUBMIT_SUBVERSION) \
                           -Wl,-compatibility_version -Wl,1
    endif
    # extra options to override libtool defaults 
	LLVMLibsOptions += -F/System/Library/Frameworks -F/System/Library/PrivateFrameworks
	LLVMLibsOptions += -framework Foundation -framework CoreFoundation
	LLVMLibsOptions += -framework CoreServices -framework Carbon -framework Security
	LLVMLibsOptions += -framework DebugSymbols $(PYTHON_BUILD_FLAGS) -lobjc
	LLVMLibsOptions += -lxml2
	ifneq ($(EXPORTED_SYMBOL_FILE),)
		LLVMLibsOptions += -Wl,-exported_symbols_list -Wl,"$(EXPORTED_SYMBOL_FILE)"
	endif
    # Mac OS X 10.4 and earlier tools do not allow a second -install_name on command line
    DARWIN_VERS := $(shell echo $(TARGET_TRIPLE) | sed 's/.*darwin\([0-9]*\).*/\1/')
    ifneq ($(DARWIN_VERS),8)
       LLVMLibsOptions  += -Wl,-install_name \
                           -Wl,"@executable_path/../lib/lib$(LIBRARYNAME)$(SHLIBEXT)"
    endif
endif

ifeq ($(HOST_OS),Linux)
    # Include everything from the .a's into the shared library.
    ProjLibsOptions := -Wl,--whole-archive $(ProjLibsOptions) \
                       -Wl,--no-whole-archive
    # Don't allow unresolved symbols.
    LLVMLibsOptions += -Wl,--no-undefined
    # Link in python
    LLVMLibsOptions += $(PYTHON_BUILD_FLAGS) -lrt
endif

ifeq ($(HOST_OS),FreeBSD)
    # Include everything from the .a's into the shared library.
    ProjLibsOptions := -Wl,--whole-archive $(ProjLibsOptions) \
                       -Wl,--no-whole-archive
    # Allow unresolved symbols.
    LLVMLibsOptions += -Wl,--allow-shlib-undefined
    # Link in python
    LLVMLibsOptions += $(PYTHON_BUILD_FLAGS) -lrt -L/usr/local/lib -lexecinfo
endif
