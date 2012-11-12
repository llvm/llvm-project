CLANG_LEVEL := ../../..
TD_SRC_DIR = $(PROJ_SRC_DIR)/../Basic
BUILT_SOURCES = Attrs.inc AttrImpl.inc StmtNodes.inc DeclNodes.inc \
                CommentNodes.inc CommentHTMLTags.inc \
                CommentHTMLTagsProperties.inc CommentCommandInfo.inc

TABLEGEN_INC_FILES_COMMON = 1

include $(CLANG_LEVEL)/Makefile

$(ObjDir)/Attrs.inc.tmp : $(TD_SRC_DIR)/Attr.td $(CLANG_TBLGEN) \
                              $(ObjDir)/.dir
	$(Echo) "Building Clang attribute classes with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-attr-classes -o $(call SYSPATH, $@) \
		-I $(PROJ_SRC_DIR)/../../ $<

$(ObjDir)/AttrImpl.inc.tmp : $(TD_SRC_DIR)/Attr.td $(CLANG_TBLGEN) \
                              $(ObjDir)/.dir
	$(Echo) "Building Clang attribute implementations with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-attr-impl -o $(call SYSPATH, $@) \
		-I $(PROJ_SRC_DIR)/../../ $<

$(ObjDir)/StmtNodes.inc.tmp : $(TD_SRC_DIR)/StmtNodes.td $(CLANG_TBLGEN) \
                              $(ObjDir)/.dir
	$(Echo) "Building Clang statement node tables with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-stmt-nodes -o $(call SYSPATH, $@) $<

$(ObjDir)/DeclNodes.inc.tmp : $(TD_SRC_DIR)/DeclNodes.td $(CLANG_TBLGEN) \
                              $(ObjDir)/.dir
	$(Echo) "Building Clang declaration node tables with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-decl-nodes -o $(call SYSPATH, $@) $<

$(ObjDir)/CommentNodes.inc.tmp : $(TD_SRC_DIR)/CommentNodes.td $(CLANG_TBLGEN) \
                              $(ObjDir)/.dir
	$(Echo) "Building Clang comment node tables with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-comment-nodes -o $(call SYSPATH, $@) $<

$(ObjDir)/CommentHTMLTags.inc.tmp : $(PROJ_SRC_DIR)/CommentHTMLTags.td $(CLANG_TBLGEN) \
                              $(ObjDir)/.dir
	$(Echo) "Building Clang comment HTML tag matchers with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-comment-html-tags -o $(call SYSPATH, $@) $<

$(ObjDir)/CommentHTMLTagsProperties.inc.tmp : $(PROJ_SRC_DIR)/CommentHTMLTags.td \
                                              $(CLANG_TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building Clang comment HTML tag properties with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-comment-html-tags-properties -o $(call SYSPATH, $@) $<

$(ObjDir)/CommentCommandInfo.inc.tmp : $(PROJ_SRC_DIR)/CommentCommands.td \
                                              $(CLANG_TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building Clang comment command info with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-comment-command-info -o $(call SYSPATH, $@) $<

