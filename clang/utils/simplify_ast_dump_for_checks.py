#!/usr/bin/env python
"""
A utility to simplify AST dumps to a format that is compatible with FileChecks.

The tool tries to only keep essential information from AST dumps to make them
easier to read and complete while keeping them complete enough to match against.

Usage: clang -Xclang -ast-dump foo.c | python simplify_ast_dump_for_checks.py
"""

import sys

def char_iter(stream):
	contents = stream.read(4000)
	while contents:
		for c in contents:
			yield c
		contents = stream.read(4000)

def parse_indent(iter, nextval):
	indent = ""
	while nextval in "`-| ":
		indent += nextval
		nextval = next(iter)
	return (nextval, indent)

def parse_spaces(iter, nextval):
	r = ''
	while nextval.isspace() or nextval == ':':
		r += nextval
		nextval = next(iter)
	return (nextval, r)

def parse_tok(iter, nextval):
	tok = ""
	while not nextval.isspace():
		tok += nextval
		nextval = next(iter)
	return (nextval, tok)

def parse_sloc(iter, nextval):
	assert nextval == '<'
	result = nextval
	nextval = next(iter)
	depth = 1
	while depth != 0:
		if nextval == '<':
			depth += 1
		elif nextval == '>':
			depth -= 1
		result += nextval
		nextval = next(iter)
	return (nextval, result)
	
def parse_quoted(iter, nextval):
	assert nextval == "'"
	result = nextval
	nextval = next(iter)
	depth = 1
	while depth != 0:
		result += nextval
		if nextval == "'":
			depth -= 1
		nextval = next(iter)
	return (nextval, result)

def parse_any_tok(iter, nextval):
	if nextval == "'":
		return parse_quoted(iter, nextval)
	elif nextval == '<':
		return parse_sloc(iter, nextval)
	else:
		return parse_tok(iter, nextval)

class AstNode(object):
	def __init__(self, indent, kind, addr, *whatev):
		self.depth = len(indent) / 2
		self.indent = indent
		self.kind = kind
		self.addr = int(addr, 0)
		self.remainder = whatev
	
	def __repr__(self):
		fields = ", ".join(repr(x) for x in (self.depth, self.kind, self.addr) + self.remainder)
		return "%s(%s)" % (type(self).__name__, fields)
	
	@classmethod
	def new(cls, payload):
		try:
			payload = list(payload)
			if payload[1] == "array_filler:":
				return ArrayFillerNode(cls, *payload)
			if payload[1].endswith("VarDecl"):
				# move "used" marker at the end for simplicity
				try:
					used_idx = payload.index("used")
				except ValueError:
					pass
				else:
					payload.append(payload[used_idx])
					del payload[used_idx]
			if payload[1].endswith("CastExpr"):
				# remove part_of_explicit_cast for simplicity
				try:
					marker_idx = payload.index("part_of_explicit_cast")
				except ValueError:
					pass
				else:
					del payload[marker_idx]
			if payload[1].endswith("VarDecl"):
				return VarDeclNode(*payload)
			if payload[1] == "FunctionDecl":
				return FuncDeclNode(*payload)
			if payload[1] == "MemberExpr":
				return MemberNode(*payload)
		except TypeError:
			pass
		
		return cls(*payload)

	@classmethod
	def parse_line(cls, iter, nextval):
		nextval, indent = parse_indent(iter, nextval)
		nextval, node_name = parse_any_tok(iter, nextval)
		tokens = [indent, node_name]
		while nextval != '\n':
			nextval, s = parse_spaces(iter, nextval)
			nextval, tok = parse_any_tok(iter, nextval)
			if s == ':':
				tokens[-1] += ':' + tok
			else:
				tokens.append(tok)
		
		if len(tokens) == 2 and tokens[-1] == "<<<NULL>>>":
			tokens.append('0')
		return (next(iter, None), cls.new(tokens))

	@classmethod
	def parse_nodes(cls, char_iter):
		nextval = next(char_iter, None)
		while nextval:
			nextval, node = cls.parse_line(char_iter, nextval)
			yield node

class VarDeclNode(AstNode):
	def __init__(self, indent, kind, addr, begin, end, name, type, *whatev):
		super(VarDeclNode, self).__init__(indent, kind, addr, *whatev)
		self.begin = begin
		self.end = end
		self.name = name
		self.type = type

class MemberNode(AstNode):
	def __init__(self, *fields):
		super(MemberNode, self).__init__(*fields)
		for f in fields:
			if f.startswith(".") or f.startswith("->"):
				self.member = f

class FuncDeclNode(AstNode):
	def __init__(self, indent, kind, addr, begin, paren, *remainder):
		type = remainder[-1]
		name = remainder[-2]
		super(FuncDeclNode, self).__init__(indent, kind, addr, *remainder[:-2])
		self.type = type
		self.name = name

class ArrayFillerNode(object):
	def __init__(self, cls, *args):
		payload = list(args)
		del payload[1]
		self.inner = cls.new(payload)
	
	def __getattr__(self, attr):
		return getattr(self.inner, attr)

def main():
	all_names = set()
	ptr_to_name = {}

	def name_for_ptr(template, ptr):
		name = template
		i = 0
		while name in all_names:
			i += 1
			name = "%s_%i" % (template, i)
		all_names.add(name)
		ptr_to_name[ptr] = name
		return name
	
	def binds_ove(node):
		if node.kind == "MaterializeSequenceExpr":
			return node.remainder[-1] == "<Bind>"
		return node.kind == "BoundsCheckExpr"

	stack = []
	skip_to_depth = None
	for node in AstNode.parse_nodes(char_iter(sys.stdin)):
		while stack and stack[-1].depth >= node.depth:
			del stack[-1]
		stack.append(node)

		may_not_skip_ove = False

		if skip_to_depth is not None:
			if node.depth > skip_to_depth:
				if node.kind == "OpaqueValueExpr" and not node.addr in ptr_to_name:
					may_not_skip_ove = True
				else:
					continue
		if not may_not_skip_ove:
			skip_to_depth = None
		if 'implicit' in node.remainder:
			skip_to_depth = node.depth
			continue
	
		if node.kind == "<<<NULL>>>":
			continue

		sys.stdout.write(node.indent)
		if isinstance(node, ArrayFillerNode):
			sys.stdout.write("array_filler: ")
			node = node.inner
		sys.stdout.write(node.kind)
	
		# what else should we print for this node?
		if isinstance(node, VarDeclNode):
			name = name_for_ptr("var_" + node.name, node.addr)
			sys.stdout.write(" [[%s:0x[^ ]+]]" % name)
		
		if isinstance(node, MemberNode):
			sys.stdout.write(" {{.+}} %s" % node.member)
	
		if isinstance(node, FuncDeclNode):
			name = name_for_ptr("func_" + node.name, node.addr)
			sys.stdout.write(" [[%s:0x[^ ]+]]" % name)
			sys.stdout.write(" {{.+}} %s" % node.name)
		
		if node.kind == "MaterializeSequenceExpr":
			sys.stdout.write(" {{.+}} %s" % node.remainder[-1])
	
		if node.kind == "DeclRefExpr":
			sys.stdout.write(" {{.+}}")
			for v in node.remainder:
				if v.startswith("0x"):
					p = int(v, 0)
					if p in ptr_to_name:
						sys.stdout.write(" [[%s]]" % ptr_to_name[p])
	
		if node.kind == "CompoundAssignOperator":
			sys.stdout.write(" {{.+}} %s %s" % node.remainder[-4:-2])
		elif node.kind.endswith("CastExpr") or node.kind.endswith("Operator"):
			sys.stdout.write(" {{.+}} %s %s" % node.remainder[-2:])
	
		if node.kind in ("IntegerLiteral", "BoundsSafetyPointerPromotionExpr", "GetBoundExpr", "LabelStmt"):
			sys.stdout.write(" {{.+}}")
			sys.stdout.write(" " + node.remainder[-1])
		
		if node.kind == "BoundsCheckExpr":
			sys.stdout.write(" {{.+}} %s" % node.remainder[-1])
	
		if node.kind == "PredefinedBoundsCheckExpr":
			sys.stdout.write(" {{.+}} %s %s" % node.remainder[-2:])

		if node.kind == "OpaqueValueExpr":
			if node.addr in ptr_to_name:
				sys.stdout.write(" [[%s]]" % ptr_to_name[node.addr])
			else:
				name = name_for_ptr("ove", node.addr)
				sys.stdout.write(" [[%s:0x[^ ]+]]" % name)
			if not binds_ove(stack[-2]):
				sys.stdout.write(" {{.*}} " + node.remainder[-1])
				if not may_not_skip_ove:
					skip_to_depth = node.depth
	
		sys.stdout.write("\n")

if __name__ == "__main__":
	sys.exit(main())
