from clang.cindex import CursorKind

def test_name():
    assert CursorKind.UNEXPOSED_DECL.name is 'UNEXPOSED_DECL'

def test_get_all_kinds():
    kinds = CursorKind.get_all_kinds()
    assert CursorKind.UNEXPOSED_DECL in kinds
    assert CursorKind.TRANSLATION_UNIT in kinds
    assert CursorKind.VARIABLE_REF in kinds
    assert CursorKind.LAMBDA_EXPR in kinds
    assert CursorKind.OBJ_BOOL_LITERAL_EXPR in kinds
    assert CursorKind.OBJ_SELF_EXPR in kinds
    assert CursorKind.MS_ASM_STMT in kinds
    assert CursorKind.MODULE_IMPORT_DECL in kinds

def test_kind_groups():
    """Check that every kind classifies to exactly one group."""

    assert CursorKind.UNEXPOSED_DECL.is_declaration()
    assert CursorKind.TYPE_REF.is_reference()
    assert CursorKind.DECL_REF_EXPR.is_expression()
    assert CursorKind.UNEXPOSED_STMT.is_statement()
    assert CursorKind.INVALID_FILE.is_invalid()

    assert CursorKind.TRANSLATION_UNIT.is_translation_unit()
    assert not CursorKind.TYPE_REF.is_translation_unit()

    assert CursorKind.PREPROCESSING_DIRECTIVE.is_preprocessing()
    assert not CursorKind.TYPE_REF.is_preprocessing()

    assert CursorKind.UNEXPOSED_DECL.is_unexposed()
    assert not CursorKind.TYPE_REF.is_unexposed()

    for k in CursorKind.get_all_kinds():
        group = [n for n in ('is_declaration', 'is_reference', 'is_expression',
                             'is_statement', 'is_invalid', 'is_attribute')
                 if getattr(k, n)()]

        if k in (   CursorKind.TRANSLATION_UNIT,
                    CursorKind.MACRO_DEFINITION,
                    CursorKind.MACRO_INSTANTIATION,
                    CursorKind.INCLUSION_DIRECTIVE,
                    CursorKind.PREPROCESSING_DIRECTIVE):
            assert len(group) == 0
        else:
            assert len(group) == 1
