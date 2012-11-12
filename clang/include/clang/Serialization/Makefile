CLANG_LEVEL := ../../..
TD_SRC_DIR = $(PROJ_SRC_DIR)/../Basic
BUILT_SOURCES = AttrPCHRead.inc AttrPCHWrite.inc

TABLEGEN_INC_FILES_COMMON = 1

include $(CLANG_LEVEL)/Makefile

$(ObjDir)/AttrPCHRead.inc.tmp : $(TD_SRC_DIR)/Attr.td $(CLANG_TBLGEN) \
                              $(ObjDir)/.dir
	$(Echo) "Building Clang PCH reader with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-attr-pch-read -o $(call SYSPATH, $@) \
		-I $(PROJ_SRC_DIR)/../../ $<

$(ObjDir)/AttrPCHWrite.inc.tmp : $(TD_SRC_DIR)/Attr.td $(CLANG_TBLGEN) \
                              $(ObjDir)/.dir
	$(Echo) "Building Clang PCH writer with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-attr-pch-write -o $(call SYSPATH, $@) \
		-I $(PROJ_SRC_DIR)/../../ $<
