class AttrSubject;

class Decl<bit abstract = 0> : AttrSubject {
  bit Abstract = abstract;
}

class DDecl<Decl base, bit abstract = 0> : Decl<abstract> {
  Decl Base = base;
}

class DeclContext { }

def TranslationUnit : Decl, DeclContext;
def Named : Decl<1>;
  def Namespace : DDecl<Named>, DeclContext;
  def UsingDirective : DDecl<Named>;
  def NamespaceAlias : DDecl<Named>;
  def Label : DDecl<Named>;
  def Type : DDecl<Named, 1>;
    def TypedefName : DDecl<Type, 1>;
      def Typedef : DDecl<TypedefName>;
      def TypeAlias : DDecl<TypedefName>;
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
    def Declarator : DDecl<Value, 1>;
      def Function : DDecl<Declarator>, DeclContext;
        def CXXMethod : DDecl<Function>;
          def CXXConstructor : DDecl<CXXMethod>;
          def CXXDestructor : DDecl<CXXMethod>;
          def CXXConversion : DDecl<CXXMethod>;
      def Field : DDecl<Declarator>;
        def ObjCIvar : DDecl<Field>;
        def ObjCAtDefsField : DDecl<Field>;
      def Var : DDecl<Declarator>;
        def ImplicitParam : DDecl<Var>;
        def ParmVar : DDecl<Var>;
      def NonTypeTemplateParm : DDecl<Declarator>;
  def Template : DDecl<Named, 1>;
    def RedeclarableTemplate : DDecl<Template, 1>;
      def FunctionTemplate : DDecl<RedeclarableTemplate>;
      def ClassTemplate : DDecl<RedeclarableTemplate>;
      def TypeAliasTemplate : DDecl<RedeclarableTemplate>;
    def TemplateTemplateParm : DDecl<Template>;
  def Using : DDecl<Named>;
  def UsingShadow : DDecl<Named>;
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
def ObjCPropertyImpl : Decl;
def FileScopeAsm : Decl;
def AccessSpec : Decl;
def Friend : Decl;
def FriendTemplate : Decl;
def StaticAssert : Decl;
def Block : Decl, DeclContext;
def ClassScopeFunctionSpecialization : Decl;
def Import : Decl;

