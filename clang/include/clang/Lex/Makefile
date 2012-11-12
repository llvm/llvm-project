CLANG_LEVEL := ../../..
TD_SRC_DIR = $(PROJ_SRC_DIR)/../Basic
BUILT_SOURCES = AttrSpellings.inc 

TABLEGEN_INC_FILES_COMMON = 1

include $(CLANG_LEVEL)/Makefile

$(ObjDir)/AttrSpellings.inc.tmp : $(TD_SRC_DIR)/Attr.td $(CLANG_TBLGEN) \
                                  $(ObjDir)/.dir
	$(Echo) "Building Clang attribute spellings with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-attr-spelling-list -o $(call SYSPATH, $@) \
		-I $(PROJ_SRC_DIR)/../../ $<
