/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Tables and code for Clang's builtin templates                              *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#if defined(CREATE_BUILTIN_TEMPLATE_PARAMETER_LIST)
case BTK__builtin_common_type: {
 auto *Parm1 = TemplateTypeParmDecl::Create(C, DC, SourceLocation(), SourceLocation(), 1, 0, /*Id=*/nullptr, /*Typename=*/false, true);
 auto *TPL2 = TemplateParameterList::Create(C, SourceLocation(), SourceLocation(), {Parm1}, SourceLocation(), nullptr);
 auto *Parm0 = TemplateTemplateParmDecl::Create(C, DC, SourceLocation(), 0, 0, /*ParameterPack=*/false, /*Id=*/nullptr, /*Typename=*/false, TPL2);
 auto *Parm4 = TemplateTypeParmDecl::Create(C, DC, SourceLocation(), SourceLocation(), 1, 0, /*Id=*/nullptr, /*Typename=*/false, false);
 auto *TPL5 = TemplateParameterList::Create(C, SourceLocation(), SourceLocation(), {Parm4}, SourceLocation(), nullptr);
 auto *Parm3 = TemplateTemplateParmDecl::Create(C, DC, SourceLocation(), 0, 1, /*ParameterPack=*/false, /*Id=*/nullptr, /*Typename=*/false, TPL5);
 auto *Parm6 = TemplateTypeParmDecl::Create(C, DC, SourceLocation(), SourceLocation(), 0, 2, /*Id=*/nullptr, /*Typename=*/false, false);
 auto *Parm7 = TemplateTypeParmDecl::Create(C, DC, SourceLocation(), SourceLocation(), 0, 3, /*Id=*/nullptr, /*Typename=*/false, true);
 auto *TPL8 = TemplateParameterList::Create(C, SourceLocation(), SourceLocation(), {Parm0, Parm3, Parm6, Parm7}, SourceLocation(), nullptr);

  return TPL8;
  }
case BTK__make_integer_seq: {
 auto *Parm1 = TemplateTypeParmDecl::Create(C, DC, SourceLocation(), SourceLocation(), 1, 0, /*Id=*/nullptr, /*Typename=*/false, false);
 auto *TSI3 = C.getTrivialTypeSourceInfo(QualType(Parm1->getTypeForDecl(), 0));
 auto *Parm2 = NonTypeTemplateParmDecl::Create(C, DC, SourceLocation(), SourceLocation(), 1, 1, /*Id=*/nullptr, TSI3->getType(), true, TSI3);
 auto *TPL4 = TemplateParameterList::Create(C, SourceLocation(), SourceLocation(), {Parm1, Parm2}, SourceLocation(), nullptr);
 auto *Parm0 = TemplateTemplateParmDecl::Create(C, DC, SourceLocation(), 0, 0, /*ParameterPack=*/false, /*Id=*/nullptr, /*Typename=*/false, TPL4);
 auto *Parm5 = TemplateTypeParmDecl::Create(C, DC, SourceLocation(), SourceLocation(), 0, 1, /*Id=*/nullptr, /*Typename=*/false, false);
 auto *TSI7 = C.getTrivialTypeSourceInfo(QualType(Parm5->getTypeForDecl(), 0));
 auto *Parm6 = NonTypeTemplateParmDecl::Create(C, DC, SourceLocation(), SourceLocation(), 0, 2, /*Id=*/nullptr, TSI7->getType(), false, TSI7);
 auto *TPL8 = TemplateParameterList::Create(C, SourceLocation(), SourceLocation(), {Parm0, Parm5, Parm6}, SourceLocation(), nullptr);

  return TPL8;
  }
case BTK__type_pack_element: {
TypeSourceInfo *SizeTInfo = C.getTrivialTypeSourceInfo(C.getSizeType());
 auto *Parm0 = NonTypeTemplateParmDecl::Create(C, DC, SourceLocation(), SourceLocation(), 0, 0, /*Id=*/nullptr, SizeTInfo->getType(), /*ParameterPack=*/false, SizeTInfo);
 auto *Parm1 = TemplateTypeParmDecl::Create(C, DC, SourceLocation(), SourceLocation(), 0, 1, /*Id=*/nullptr, /*Typename=*/false, true);
 auto *TPL2 = TemplateParameterList::Create(C, SourceLocation(), SourceLocation(), {Parm0, Parm1}, SourceLocation(), nullptr);

  return TPL2;
  }
#undef CREATE_BUILTIN_TEMPLATE_PARAMETER_LIST
#else
BuiltinTemplate(__builtin_common_type)
BuiltinTemplate(__make_integer_seq)
BuiltinTemplate(__type_pack_element)
#undef BuiltinTemplate
#endif
