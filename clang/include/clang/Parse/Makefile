CLANG_LEVEL := ../../..
TD_SRC_DIR = $(PROJ_SRC_DIR)/../Basic
BUILT_SOURCES = AttrLateParsed.inc

TABLEGEN_INC_FILES_COMMON = 1

include $(CLANG_LEVEL)/Makefile

$(ObjDir)/AttrLateParsed.inc.tmp : $(TD_SRC_DIR)/Attr.td $(CLANG_TBLGEN) \
                                   $(ObjDir)/.dir
	$(Echo) "Building Clang attribute late-parsed table with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-attr-late-parsed-list -o $(call SYSPATH, $@) \
		-I $(PROJ_SRC_DIR)/../../ $<