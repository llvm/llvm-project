"""Code generator for Code Completion Model Inference.

Tool runs on the Decision Forest model defined in {model} directory.
It generates two files: {output_dir}/{filename}.h and {output_dir}/{filename}.cpp 
The generated files defines the Example class named {cpp_class} having all the features as class members.
The generated runtime provides an `Evaluate` function which can be used to score a code completion candidate.
"""

import argparse
import json
import struct


class CppClass:
    """Holds class name and names of the enclosing namespaces."""

    def __init__(self, cpp_class):
        ns_and_class = cpp_class.split("::")
        self.ns = [ns for ns in ns_and_class[0:-1] if len(ns) > 0]
        self.name = ns_and_class[-1]
        if len(self.name) == 0:
            raise ValueError("Empty class name.")

    def ns_begin(self):
        """Returns snippet for opening namespace declarations."""
        open_ns = ["namespace %s {" % ns for ns in self.ns]
        return "\n".join(open_ns)

    def ns_end(self):
        """Returns snippet for closing namespace declarations."""
        close_ns = [
            "} // namespace %s" % ns for ns in reversed(self.ns)]
        return "\n".join(close_ns)


def header_guard(filename):
    '''Returns the header guard for the generated header.'''
    return "GENERATED_DECISION_FOREST_MODEL_%s_H" % filename.upper()


def boost_node(n, label, next_label):
    """Returns code snippet for a leaf/boost node.
    Adds value of leaf to the score and jumps to the root of the next tree."""
    return "%s: Score += %s; goto %s;" % (
            label, n['score'], next_label)


def if_greater_node(n, label, next_label):
    """Returns code snippet for a if_greater node.
    Jumps to true_label if the Example feature (NUMBER) is greater than the threshold. 
    Comparing integers is much faster than comparing floats. Assuming floating points 
    are represented as IEEE 754, it order-encodes the floats to integers before comparing them.
    Control falls through if condition is evaluated to false."""
    threshold = n["threshold"]
    return "%s: if (E.%s >= %s /*%s*/) goto %s;" % (
            label, n['feature'], order_encode(threshold), threshold, next_label)


def if_member_node(n, label, next_label):
    """Returns code snippet for a if_member node.
    Jumps to true_label if the Example feature (ENUM) is present in the set of enum values 
    described in the node.
    Control falls through if condition is evaluated to false."""
    members = '|'.join([
        "BIT(%s_type::%s)" % (n['feature'], member)
        for member in n["set"]
    ])
    return "%s: if (E.%s & (%s)) goto %s;" % (
            label, n['feature'], members, next_label)


def node(n, label, next_label):
    """Returns code snippet for the node."""
    return {
        'boost': boost_node,
        'if_greater': if_greater_node,
        'if_member': if_member_node,
    }[n['operation']](n, label, next_label)


def tree(t, tree_num, node_num):
    """Returns code for inferencing a Decision Tree.
    Also returns the size of the decision tree.

    A tree starts with its label `t{tree#}`.
    A node of the tree starts with label `t{tree#}_n{node#}`.

    The tree contains two types of node: Conditional node and Leaf node.
    -   Conditional node evaluates a condition. If true, it jumps to the true node/child.
        Code is generated using pre-order traversal of the tree considering
        false node as the first child. Therefore the false node is always the
        immediately next label.
    -   Leaf node adds the value to the score and jumps to the next tree.
    """
    label = "t%d_n%d" % (tree_num, node_num)
    code = []
    if node_num == 0:
        code.append("t%d:" % tree_num)

    if t["operation"] == "boost":
        code.append(node(t, label=label, next_label="t%d" % (tree_num + 1)))
        return code, 1

    false_code, false_size = tree(
        t['else'], tree_num=tree_num, node_num=node_num+1)

    true_node_num = node_num+false_size+1
    true_label = "t%d_n%d" % (tree_num, true_node_num)

    true_code, true_size = tree(
        t['then'], tree_num=tree_num, node_num=true_node_num)

    code.append(node(t, label=label, next_label=true_label))

    return code+false_code+true_code, 1+false_size+true_size


def gen_header_code(features_json, cpp_class, filename):
    """Returns code for header declaring the inference runtime.

    Declares the Example class named {cpp_class} inside relevant namespaces.
    The Example class contains all the features as class members. This 
    class can be used to represent a code completion candidate.
    Provides `float Evaluate()` function which can be used to score the Example.
    """
    setters = []
    for f in features_json:
        feature = f["name"]
        if f["kind"] == "NUMBER":
            # Floats are order-encoded to integers for faster comparison.
            setters.append(
                "void set%s(float V) { %s = OrderEncode(V); }" % (
                    feature, feature))
        elif f["kind"] == "ENUM":
            setters.append(
                "void set%s(unsigned V) { %s = 1 << V; }" % (feature, feature))
        else:
            raise ValueError("Unhandled feature type.", f["kind"])

    # Class members represent all the features of the Example.
    class_members = ["uint32_t %s = 0;" % f['name'] for f in features_json]

    nline = "\n  "
    guard = header_guard(filename)
    return """#ifndef %s
#define %s
#include <cstdint>

%s
class %s {
public:
  %s

private:
  %s

  // Produces an integer that sorts in the same order as F.
  // That is: a < b <==> orderEncode(a) < orderEncode(b).
  static uint32_t OrderEncode(float F);
  friend float Evaluate(const %s&);
};

float Evaluate(const %s&);
%s
#endif // %s
""" % (guard, guard, cpp_class.ns_begin(), cpp_class.name, nline.join(setters),
        nline.join(class_members), cpp_class.name, cpp_class.name,
        cpp_class.ns_end(), guard)


def order_encode(v):
    i = struct.unpack('<I', struct.pack('<f', v))[0]
    TopBit = 1 << 31
    # IEEE 754 floats compare like sign-magnitude integers.
    if (i & TopBit):  # Negative float
        return (1 << 32) - i  # low half of integers, order reversed.
    return TopBit + i  # top half of integers


def evaluate_func(forest_json, cpp_class):
    """Generates code for `float Evaluate(const {Example}&)` function.
    The generated function can be used to score an Example."""
    code = "float Evaluate(const %s& E) {\n" % cpp_class.name
    lines = []
    lines.append("float Score = 0;")
    tree_num = 0
    for tree_json in forest_json:
        lines.extend(tree(tree_json, tree_num=tree_num, node_num=0)[0])
        lines.append("")
        tree_num += 1

    lines.append("t%s: // No such tree." % len(forest_json))
    lines.append("return Score;")
    code += "  " + "\n  ".join(lines)
    code += "\n}"
    return code


def gen_cpp_code(forest_json, features_json, filename, cpp_class):
    """Generates code for the .cpp file."""
    # Headers
    # Required by OrderEncode(float F).
    angled_include = [
        '#include <%s>' % h
        for h in ["cstring", "limits"]
    ]

    # Include generated header.
    qouted_headers = {filename + '.h', 'llvm/ADT/bit.h'}
    # Headers required by ENUM features used by the model.
    qouted_headers |= {f["header"]
                       for f in features_json if f["kind"] == "ENUM"}
    quoted_include = ['#include "%s"' % h for h in sorted(qouted_headers)]

    # using-decl for ENUM features.
    using_decls = "\n".join("using %s_type = %s;" % (
                                feature['name'], feature['type'])
                            for feature in features_json
                            if feature["kind"] == "ENUM")
    nl = "\n"
    return """%s

%s

#define BIT(X) (1 << X)

%s

%s

uint32_t %s::OrderEncode(float F) {
  static_assert(std::numeric_limits<float>::is_iec559, "");
  constexpr uint32_t TopBit = ~(~uint32_t{0} >> 1);

  // Get the bits of the float. Endianness is the same as for integers.
  uint32_t U = llvm::bit_cast<uint32_t>(F);
  std::memcpy(&U, &F, sizeof(U));
  // IEEE 754 floats compare like sign-magnitude integers.
  if (U & TopBit)    // Negative float.
    return 0 - U;    // Map onto the low half of integers, order reversed.
  return U + TopBit; // Positive floats map onto the high half of integers.
}

%s
%s
""" % (nl.join(angled_include), nl.join(quoted_include), cpp_class.ns_begin(),
       using_decls, cpp_class.name, evaluate_func(forest_json, cpp_class),
       cpp_class.ns_end())


def main():
    parser = argparse.ArgumentParser('DecisionForestCodegen')
    parser.add_argument('--filename', help='output file name.')
    parser.add_argument('--output_dir', help='output directory.')
    parser.add_argument('--model', help='path to model directory.')
    parser.add_argument(
        '--cpp_class',
        help='The name of the class (which may be a namespace-qualified) created in generated header.'
    )
    ns = parser.parse_args()

    output_dir = ns.output_dir
    filename = ns.filename
    header_file = "%s/%s.h" % (output_dir, filename)
    cpp_file = "%s/%s.cpp" % (output_dir, filename)
    cpp_class = CppClass(cpp_class=ns.cpp_class)

    model_file = "%s/forest.json" % ns.model
    features_file = "%s/features.json" % ns.model

    with open(features_file) as f:
        features_json = json.load(f)

    with open(model_file) as m:
        forest_json = json.load(m)

    with open(cpp_file, 'w+t') as output_cc:
        output_cc.write(
            gen_cpp_code(forest_json=forest_json,
                         features_json=features_json,
                         filename=filename,
                         cpp_class=cpp_class))

    with open(header_file, 'w+t') as output_h:
        output_h.write(gen_header_code(
            features_json=features_json, cpp_class=cpp_class, filename=filename))


if __name__ == '__main__':
    main()
