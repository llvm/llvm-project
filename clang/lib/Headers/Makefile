##===- clang/lib/Headers/Makefile --------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

CLANG_LEVEL := ../..

BUILT_SOURCES = arm_neon.h.inc
TABLEGEN_INC_FILES_COMMON = 1

include $(CLANG_LEVEL)/Makefile

CLANG_VERSION := $(word 3,$(shell grep "CLANG_VERSION " \
	$(PROJ_OBJ_DIR)/$(CLANG_LEVEL)/include/clang/Basic/Version.inc))

HeaderDir := $(PROJ_OBJ_ROOT)/$(BuildMode)/lib/clang/$(CLANG_VERSION)/include

HEADERS := $(notdir $(wildcard $(PROJ_SRC_DIR)/*.h))

OBJHEADERS := $(addprefix $(HeaderDir)/, $(HEADERS))


$(OBJHEADERS): $(HeaderDir)/%.h: $(PROJ_SRC_DIR)/%.h $(HeaderDir)/.dir $(HeaderDir)/arm_neon.h
	$(Verb) cp $< $@
	$(Echo) Copying $(notdir $<) to build dir

$(HeaderDir)/arm_neon.h: $(BUILT_SOURCES) $(HeaderDir)/.dir
	$(Verb) cp $< $@
	$(Echo) Copying $(notdir $<) to build dir

$(HeaderDir)/module.map: $(PROJ_SRC_DIR)/module.map $(HeaderDir)/.dir
	$(Verb) cp $< $@
	$(Echo) Copying $(notdir $<) to build dir


# Hook into the standard Makefile rules.
all-local:: $(OBJHEADERS) $(HeaderDir)/module.map

PROJ_headers := $(DESTDIR)$(PROJ_prefix)/lib/clang/$(CLANG_VERSION)/include

INSTHEADERS := $(addprefix $(PROJ_headers)/, $(HEADERS))
INSTHEADERS += $(PROJ_headers)/arm_neon.h

$(PROJ_headers):
	$(Verb) $(MKDIR) $@

$(INSTHEADERS): $(PROJ_headers)/%.h: $(HeaderDir)/%.h | $(PROJ_headers)
	$(Verb) $(DataInstall) $< $(PROJ_headers)
	$(Echo) Installing compiler include file: $(notdir $<)

$(PROJ_headers)/module.map: $(HeaderDir)/module.map | $(PROJ_headers)
	$(Verb) $(DataInstall) $< $(PROJ_headers)
	$(Echo) Installing compiler module map file: $(notdir $<)


install-local:: $(INSTHEADERS) $(PROJ_headers)/module.map

$(ObjDir)/arm_neon.h.inc.tmp : $(CLANG_LEVEL)/include/clang/Basic/arm_neon.td $(CLANG_TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building Clang arm_neon.h.inc with tblgen"
	$(Verb) $(ClangTableGen) -gen-arm-neon -o $(call SYSPATH, $@) $<
