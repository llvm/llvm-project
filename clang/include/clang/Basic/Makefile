CLANG_LEVEL := ../../..
BUILT_SOURCES = \
	DiagnosticAnalysisKinds.inc DiagnosticASTKinds.inc \
	DiagnosticCommentKinds.inc \
	DiagnosticCommonKinds.inc DiagnosticDriverKinds.inc \
	DiagnosticFrontendKinds.inc DiagnosticLexKinds.inc \
	DiagnosticParseKinds.inc DiagnosticSemaKinds.inc \
	DiagnosticSerializationKinds.inc \
	DiagnosticIndexName.inc DiagnosticGroups.inc AttrList.inc arm_neon.inc \
	Version.inc

TABLEGEN_INC_FILES_COMMON = 1

include $(CLANG_LEVEL)/Makefile

INPUT_TDS = $(wildcard $(PROJ_SRC_DIR)/Diagnostic*.td)

# Compute the Clang version from the LLVM version, unless specified explicitly.
ifndef CLANG_VERSION
CLANG_VERSION := $(subst svn,,$(LLVMVersion))
CLANG_VERSION := $(subst rc,,$(CLANG_VERSION))
endif

CLANG_VERSION_COMPONENTS := $(subst ., ,$(CLANG_VERSION))
CLANG_VERSION_MAJOR := $(word 1,$(CLANG_VERSION_COMPONENTS))
CLANG_VERSION_MINOR := $(word 2,$(CLANG_VERSION_COMPONENTS))
CLANG_VERSION_PATCHLEVEL := $(word 3,$(CLANG_VERSION_COMPONENTS))
ifeq ($(CLANG_VERSION_PATCHLEVEL),)
CLANG_HAS_VERSION_PATCHLEVEL := 0
else
CLANG_HAS_VERSION_PATCHLEVEL := 1
endif

$(ObjDir)/Diagnostic%Kinds.inc.tmp : Diagnostic.td $(INPUT_TDS) $(CLANG_TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building Clang $(patsubst Diagnostic%Kinds.inc.tmp,%,$(@F)) diagnostic tables with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-diags-defs -clang-component=$(patsubst Diagnostic%Kinds.inc.tmp,%,$(@F)) -o $(call SYSPATH, $@) $<

$(ObjDir)/DiagnosticIndexName.inc.tmp : Diagnostic.td $(INPUT_TDS) $(CLANG_TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building Clang diagnostic name index with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-diags-index-name -o $(call SYSPATH, $@) $<

$(ObjDir)/DiagnosticGroups.inc.tmp : Diagnostic.td DiagnosticGroups.td $(INPUT_TDS) $(CLANG_TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building Clang diagnostic groups with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-diag-groups -o $(call SYSPATH, $@) $<

$(ObjDir)/AttrList.inc.tmp : Attr.td $(CLANG_TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building Clang attribute list with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-attr-list -o $(call SYSPATH, $@) \
	  -I $(PROJ_SRC_DIR)/../.. $<

$(ObjDir)/arm_neon.inc.tmp : arm_neon.td $(CLANG_TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building Clang arm_neon.inc with tblgen"
	$(Verb) $(ClangTableGen) -gen-arm-neon-sema -o $(call SYSPATH, $@) $<

$(ObjDir)/Version.inc.tmp : Version.inc.in Makefile $(LLVM_OBJ_ROOT)/Makefile.config $(ObjDir)/.dir
	$(Echo) "Updating Clang version info."
	$(Verb)sed -e "s#@CLANG_VERSION@#$(CLANG_VERSION)#g" \
	           -e "s#@CLANG_VERSION_MAJOR@#$(CLANG_VERSION_MAJOR)#g" \
	           -e "s#@CLANG_VERSION_MINOR@#$(CLANG_VERSION_MINOR)#g" \
	           -e "s#@CLANG_VERSION_PATCHLEVEL@#$(CLANG_VERSION_PATCHLEVEL)#g" \
	           -e "s#@CLANG_HAS_VERSION_PATCHLEVEL@#$(CLANG_HAS_VERSION_PATCHLEVEL)#g" \
	           $< > $@
