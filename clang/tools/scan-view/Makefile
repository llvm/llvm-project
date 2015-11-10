##===- tools/scan-view/Makefile ----------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

CLANG_LEVEL := ../..

include $(CLANG_LEVEL)/../../Makefile.config
include $(CLANG_LEVEL)/Makefile

CLANG_INSTALL_SCANVIEW ?= 1

ifeq ($(CLANG_INSTALL_SCANVIEW), 1)
  InstallTargets := $(ToolDir)/Reporter.py \
                    $(ToolDir)/ScanView.py \
                    $(ToolDir)/scan-view \
                    $(ToolDir)/startfile.py \
                    $(ShareDir)/scan-view/FileRadar.scpt \
                    $(ShareDir)/scan-view/GetRadarVersion.scpt \
                    $(ShareDir)/scan-view/bugcatcher.ico
endif

all:: $(InstallTargets)

$(ToolDir)/%: % Makefile $(ToolDir)/.dir
	$(Echo) "Copying $(notdir $<) to the 'bin' directory..."
	$(Verb)cp $< $@
	$(Verb)chmod +x $@

$(ShareDir)/scan-view/%: Resources/% Makefile $(ShareDir)/scan-view/.dir
	$(Echo) "Copying $(notdir $<) to the 'share' directory..."
	$(Verb)cp $< $@

