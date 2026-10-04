class HasProperties;
class ASTNode : HasProperties;
class AttrSubject;

class EvaluateClass<ExprNode base, bit abstract = 0> : ASTNode, AttrSubject {
  ExprNode Base = base;
  bit Abstract = abstract;
}

def BaseNode : EvaluateClass<?, 1>;

def ExprNode : EvaluateClass<BaseNode, 1>;

def ConstantNode : EvaluateClass<ExprNode>;
def ArrayConstructorNode  : EvaluateClass<ExprNode>;
def FunctionRefNode  : EvaluateClass<ExprNode>;
def DesignatorNode  : EvaluateClass<ExprNode>;

// Operations
def OperationNode : EvaluateClass<ExprNode, 1>;
def ConvertNode  : EvaluateClass<OperationNode>;
def ParenthesesNode  : EvaluateClass<OperationNode>;
def NegateNode  : EvaluateClass<OperationNode>;
def ComplexComponentNode  : EvaluateClass<OperationNode>;
def NotNode  : EvaluateClass<OperationNode>;
def SetLengthNode  : EvaluateClass<OperationNode>;
def AddNode  : EvaluateClass<OperationNode>;
def SubtractNode  : EvaluateClass<OperationNode>;
def MultiplyNode  : EvaluateClass<OperationNode>;
def DivideNode  : EvaluateClass<OperationNode>;
def PowerNode  : EvaluateClass<OperationNode>;
def RealToIntPowerNode  : EvaluateClass<OperationNode>;
def ExtremumNode  : EvaluateClass<OperationNode>;
def ComplexConstructorNode  : EvaluateClass<OperationNode>;
def ConcatNode  : EvaluateClass<OperationNode>;
def LogicalOperationNode  : EvaluateClass<OperationNode>;
def RelationalNode  : EvaluateClass<OperationNode>;


// Other
def ImpliedDoIndexNode  : EvaluateClass<BaseNode>;
def TypeParamInquiriesNode  : EvaluateClass<BaseNode>;
def DescriptorInquiryNode  : EvaluateClass<BaseNode>;
def RankOneBoundElementNode  : EvaluateClass<BaseNode>;
def ActualArgumentNode  : EvaluateClass<BaseNode>; // Expr, ActualArgument, ConditionalArg
