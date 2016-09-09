class AttrSubject;

class Decl<bit abstract = 0> : AttrSubject {
  bit Abstract = abstract;
}

class DDecl<Decl base, bit abstract = 0> : Decl<abstract> {
  Decl Base = base;
}

class DeclContext { }

def TranslationUnit : Decl, DeclContext;
def PragmaComment : Decl;
def PragmaDetectMismatch : Decl;
def ExternCContext : Decl, DeclContext;
def Named : Decl<1>;
  def Namespace : DDecl<Named>, DeclContext;
  def UsingDirective : DDecl<Named>;
  def NamespaceAlias : DDecl<Named>;
  def Label : DDecl<Named>;
  def Type : DDecl<Named, 1>;
    def TypedefName : DDecl<Type, 1>;
      def Typedef : DDecl<TypedefName>;
      def TypeAlias : DDecl<TypedefName>;
      def ObjCTypeParam : DDecl<TypedefName>;
    def UnresolvedUsingTypename : DDecl<Type>;
    def Tag : DDecl<Type, 1>, DeclContext;
      def Enum : DDecl<Tag>;
      def Record : DDecl<Tag>;
        def CXXRecord : DDecl<Record>;
          def ClassTemplateSpecialization : DDecl<CXXRecord>;
            def ClassTemplatePartialSpecialization
              : DDecl<ClassTemplateSpecialization>;
    def TemplateTypeParm : DDecl<Type>;
  def Value : DDecl<Named, 1>;
    def EnumConstant : DDecl<Value>;
    def UnresolvedUsingValue : DDecl<Value>;
    def IndirectField : DDecl<Value>;
    def Binding : DDecl<Value>;
    def OMPDeclareReduction : DDecl<Value>, DeclContext;
    def Declarator : DDecl<Value, 1>;
      def Field : DDecl<Declarator>;
        def ObjCIvar : DDecl<Field>;
        def ObjCAtDefsField : DDecl<Field>;
      def MSProperty : DDecl<Declarator>;
      def Function : DDecl<Declarator>, DeclContext;
        def CXXMethod : DDecl<Function>;
          def CXXConstructor : DDecl<CXXMethod>;
          def CXXDestructor : DDecl<CXXMethod>;
          def CXXConversion : DDecl<CXXMethod>;
      def Var : DDecl<Declarator>;
        def VarTemplateSpecialization : DDecl<Var>;
          def VarTemplatePartialSpecialization
            : DDecl<VarTemplateSpecialization>;
        def ImplicitParam : DDecl<Var>;
        def ParmVar : DDecl<Var>;
        def Decomposition : DDecl<Var>;
        def OMPCapturedExpr : DDecl<Var>;
      def NonTypeTemplateParm : DDecl<Declarator>;
  def Template : DDecl<Named, 1>;
    def RedeclarableTemplate : DDecl<Template, 1>;
      def FunctionTemplate : DDecl<RedeclarableTemplate>;
      def ClassTemplate : DDecl<RedeclarableTemplate>;
      def VarTemplate : DDecl<RedeclarableTemplate>;
      def TypeAliasTemplate : DDecl<RedeclarableTemplate>;
    def TemplateTemplateParm : DDecl<Template>;
    def BuiltinTemplate : DDecl<Template>;
  def Using : DDecl<Named>;
  def UsingShadow : DDecl<Named>;
    def ConstructorUsingShadow : DDecl<UsingShadow>;
  def ObjCMethod : DDecl<Named>, DeclContext;
  def ObjCContainer : DDecl<Named, 1>, DeclContext;
    def ObjCCategory : DDecl<ObjCContainer>;
    def ObjCProtocol : DDecl<ObjCContainer>;
    def ObjCInterface : DDecl<ObjCContainer>;
    def ObjCImpl : DDecl<ObjCContainer, 1>;
      def ObjCCategoryImpl : DDecl<ObjCImpl>;
      def ObjCImplementation : DDecl<ObjCImpl>;
  def ObjCProperty : DDecl<Named>;
  def ObjCCompatibleAlias : DDecl<Named>;
def LinkageSpec : Decl, DeclContext;
def Export : Decl, DeclContext;
def ObjCPropertyImpl : Decl;
def FileScopeAsm : Decl;
def AccessSpec : Decl;
def Friend : Decl;
def FriendTemplate : Decl;
def StaticAssert : Decl;
def Block : Decl, DeclContext;
def Captured : Decl, DeclContext;
def ClassScopeFunctionSpecialization : Decl;
def Import : Decl;
def OMPThreadPrivate : Decl;
def Empty : Decl;

