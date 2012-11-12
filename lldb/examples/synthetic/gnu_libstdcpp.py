import re
import lldb.formatters.Logger

# C++ STL formatters for LLDB
# These formatters are based upon the version of the GNU libstdc++
# as it ships with Mac OS X 10.6.8 thru 10.8.0
# You are encouraged to look at the STL implementation for your platform
# before relying on these formatters to do the right thing for your setup

class StdListSynthProvider:

	def __init__(self, valobj, dict):
		logger = lldb.formatters.Logger.Logger()
		self.valobj = valobj
		self.count = None
		logger >> "Providing synthetic children for a map named " + str(valobj.GetName())

	def next_node(self,node):
		logger = lldb.formatters.Logger.Logger()
		return node.GetChildMemberWithName('_M_next')

	def is_valid(self,node):
		logger = lldb.formatters.Logger.Logger()
		return self.value(self.next_node(node)) != self.node_address

	def value(self,node):
		logger = lldb.formatters.Logger.Logger()
		return node.GetValueAsUnsigned()

	# Floyd's cyle-finding algorithm
	# try to detect if this list has a loop
	def has_loop(self):
		global _list_uses_loop_detector
		logger = lldb.formatters.Logger.Logger()
		if _list_uses_loop_detector == False:
			logger >> "Asked not to use loop detection"
			return False
		slow = self.next
		fast1 = self.next
		fast2 = self.next
		while self.is_valid(slow):
			slow_value = self.value(slow)
			fast1 = self.next_node(fast2)
			fast2 = self.next_node(fast1)
			if self.value(fast1) == slow_value or self.value(fast2) == slow_value:
				return True
			slow = self.next_node(slow)
		return False

	def num_children(self):
		global _list_capping_size
		logger = lldb.formatters.Logger.Logger()
		if self.count == None:
			self.count = self.num_children_impl()
			if self.count > _list_capping_size:
				self.count = _list_capping_size
		return self.count

	def num_children_impl(self):
		logger = lldb.formatters.Logger.Logger()
		global _list_capping_size
		try:
			next_val = self.next.GetValueAsUnsigned(0)
			prev_val = self.prev.GetValueAsUnsigned(0)
			# After a std::list has been initialized, both next and prev will be non-NULL
			if next_val == 0 or prev_val == 0:
				return 0
			if next_val == self.node_address:
				return 0
			if next_val == prev_val:
				return 1
			if self.has_loop():
				return 0
			size = 2
			current = self.next
			while current.GetChildMemberWithName('_M_next').GetValueAsUnsigned(0) != self.node_address:
				size = size + 1
				current = current.GetChildMemberWithName('_M_next')
				if size > _list_capping_size:
					return _list_capping_size
			return (size - 1)
		except:
			return 0;

	def get_child_index(self,name):
		logger = lldb.formatters.Logger.Logger()
		try:
			return int(name.lstrip('[').rstrip(']'))
		except:
			return -1

	def get_child_at_index(self,index):
		logger = lldb.formatters.Logger.Logger()
		logger >> "Fetching child " + str(index)
		if index < 0:
			return None;
		if index >= self.num_children():
			return None;
		try:
			offset = index
			current = self.next
			while offset > 0:
				current = current.GetChildMemberWithName('_M_next')
				offset = offset - 1
			return current.CreateChildAtOffset('['+str(index)+']',2*current.GetType().GetByteSize(),self.data_type)
		except:
			return None

	def extract_type(self):
		logger = lldb.formatters.Logger.Logger()
		list_type = self.valobj.GetType().GetUnqualifiedType()
		if list_type.IsReferenceType():
			list_type = list_type.GetDereferencedType()
		if list_type.GetNumberOfTemplateArguments() > 0:
			data_type = list_type.GetTemplateArgumentType(0)
		else:
			data_type = None
		return data_type

	def update(self):
		logger = lldb.formatters.Logger.Logger()
		# preemptively setting this to None - we might end up changing our mind later
		self.count = None
		try:
			impl = self.valobj.GetChildMemberWithName('_M_impl')
			node = impl.GetChildMemberWithName('_M_node')
			self.node_address = self.valobj.AddressOf().GetValueAsUnsigned(0)
			self.next = node.GetChildMemberWithName('_M_next')
			self.prev = node.GetChildMemberWithName('_M_prev')
			self.data_type = self.extract_type()
			self.data_size = self.data_type.GetByteSize()
		except:
			pass

	def has_children(self):
		logger = lldb.formatters.Logger.Logger()
		if self.count == None:
			self.update ()
			try:
				next_val = self.next.GetValueAsUnsigned(0)
				prev_val = self.prev.GetValueAsUnsigned(0)
				if next_val == 0 or prev_val == 0:
					return False
				if next_val == self.node_address:
					return False
				# skip all the advanced logic to detect the exact count of children
				# in the interest of speed from this point on, we MIGHT have children
				# our loop detection logic will still make nothing show up :)
				return True
			except:
				return False
		if self.count == 0:
			return False
		return True

class StdVectorSynthProvider:

	def __init__(self, valobj, dict):
		logger = lldb.formatters.Logger.Logger()
		self.count = None
		self.valobj = valobj
		logger >> "Providing synthetic children for a map named " + str(valobj.GetName())

	def num_children(self):
		logger = lldb.formatters.Logger.Logger()
		if self.count == None:
			self.count = self.num_children_impl()
		return self.count

	def is_valid_pointer(ptr,process):
		logger = lldb.formatters.Logger.Logger()
		error = lldb.SBError()
		process.ReadMemory(ptr,1,error)
		return False if error.Fail() else True

	def num_children_impl(self):
		logger = lldb.formatters.Logger.Logger()
		try:
			start_val = self.start.GetValueAsUnsigned(0)
			finish_val = self.finish.GetValueAsUnsigned(0)
			end_val  = self.end.GetValueAsUnsigned(0)
			# Before a vector has been constructed, it will contain bad values
			# so we really need to be careful about the length we return since
			# unitialized data can cause us to return a huge number. We need
			# to also check for any of the start, finish or end of storage values
			# being zero (NULL). If any are, then this vector has not been 
			# initialized yet and we should return zero

			# Make sure nothing is NULL
			if start_val == 0 or finish_val == 0 or end_val == 0:
				return 0
			# Make sure start is less than finish
			if start_val >= finish_val:
				return 0
			# Make sure finish is less than or equal to end of storage
			if finish_val > end_val:
				return 0

			# if we have a struct (or other data type that the compiler pads to native word size)
			# this check might fail, unless the sizeof() we get is itself incremented to take the
			# padding bytes into account - on current clang it looks like this is the case
			num_children = (finish_val-start_val)
			if (num_children % self.data_size) != 0:
				return 0
			else:
				num_children = num_children/self.data_size
			return num_children
		except:
			return 0;

	def get_child_index(self,name):
		logger = lldb.formatters.Logger.Logger()
		try:
			return int(name.lstrip('[').rstrip(']'))
		except:
			return -1

	def get_child_at_index(self,index):
		logger = lldb.formatters.Logger.Logger()
		logger >> "Retrieving child " + str(index)
		if index < 0:
			return None;
		if index >= self.num_children():
			return None;
		try:
			offset = index * self.data_size
			return self.start.CreateChildAtOffset('['+str(index)+']',offset,self.data_type)
		except:
			return None

	def update(self):
		logger = lldb.formatters.Logger.Logger()
		# preemptively setting this to None - we might end up changing our mind later
		self.count = None
		try:
			impl = self.valobj.GetChildMemberWithName('_M_impl')
			self.start = impl.GetChildMemberWithName('_M_start')
			self.finish = impl.GetChildMemberWithName('_M_finish')
			self.end = impl.GetChildMemberWithName('_M_end_of_storage')
			self.data_type = self.start.GetType().GetPointeeType()
			self.data_size = self.data_type.GetByteSize()
			# if any of these objects is invalid, it means there is no point in trying to fetch anything
			if self.start.IsValid() and self.finish.IsValid() and self.end.IsValid() and self.data_type.IsValid():
				self.count = None
			else:
				self.count = 0
		except:
			pass
			

	def has_children(self):
		return self.num_children() > 0


class StdMapSynthProvider:

	def __init__(self, valobj, dict):
		logger = lldb.formatters.Logger.Logger()
		self.valobj = valobj;
		self.count = None
		logger >> "Providing synthetic children for a map named " + str(valobj.GetName())
		
	# we need this function as a temporary workaround for rdar://problem/10801549
	# which prevents us from extracting the std::pair<K,V> SBType out of the template
	# arguments for _Rep_Type _M_t in the map itself - because we have to make up the
	# typename and then find it, we may hit the situation were std::string has multiple
	# names but only one is actually referenced in the debug information. hence, we need
	# to replace the longer versions of std::string with the shorter one in order to be able
	# to find the type name
	def fixup_class_name(self, class_name):
		logger = lldb.formatters.Logger.Logger()
		if class_name == 'std::basic_string<char, std::char_traits<char>, std::allocator<char> >':
			return 'std::basic_string<char>',True
		if class_name == 'basic_string<char, std::char_traits<char>, std::allocator<char> >':
			return 'std::basic_string<char>',True
		if class_name == 'std::basic_string<char, std::char_traits<char>, std::allocator<char> >':
			return 'std::basic_string<char>',True
		if class_name == 'basic_string<char, std::char_traits<char>, std::allocator<char> >':
			return 'std::basic_string<char>',True
		return class_name,False

	def update(self):
		logger = lldb.formatters.Logger.Logger()
		# preemptively setting this to None - we might end up changing our mind later
		self.count = None
		try:
			# we will set this to True if we find out that discovering a node in the map takes more steps than the overall size of the RB tree
			# if this gets set to True, then we will merrily return None for any child from that moment on
			self.garbage = False
			self.Mt = self.valobj.GetChildMemberWithName('_M_t')
			self.Mimpl = self.Mt.GetChildMemberWithName('_M_impl')
			self.Mheader = self.Mimpl.GetChildMemberWithName('_M_header')
			
			map_type = self.valobj.GetType()
			if map_type.IsReferenceType():
				logger >> "Dereferencing type"
				map_type = map_type.GetDereferencedType()
			
			map_arg_0 = str(map_type.GetTemplateArgumentType(0).GetName())
			map_arg_1 = str(map_type.GetTemplateArgumentType(1).GetName())
			
			logger >> "map has args " + str(map_arg_0) + " and " + str(map_arg_1)
			
			map_arg_0,fixed_0 = self.fixup_class_name(map_arg_0)
			map_arg_1,fixed_1 = self.fixup_class_name(map_arg_1)
			
			logger >> "arg_0 has become: " + str(map_arg_0) + " (fixed: " + str(fixed_0) + ")"
			logger >> "arg_1 has become: " + str(map_arg_1) + " (fixed: " + str(fixed_1) + ")"
			
			# HACK: this is related to the above issue with the typename for std::string
			# being shortened by clang - the changes to typename display and searching to honor
			# namespaces make it so that we go looking for std::pair<const std::basic_string<char>, ...>
			# but when we find a type for this, we then compare it against the fully-qualified
			# std::pair<const std::basic_string<char, std::char_traits... and of course fail
			# the way to bypass this problem is to avoid using the std:: prefix in this specific case
			if fixed_0 or fixed_1:
				map_arg_type = "pair<const " + map_arg_0 + ", " + map_arg_1
			else:
				map_arg_type = "std::pair<const " + map_arg_0 + ", " + map_arg_1
			
			if map_arg_1[-1] == '>':
				map_arg_type = map_arg_type + " >"
			else:
				map_arg_type = map_arg_type + ">"
			
			logger >> "final contents datatype is: " + str(map_arg_type)
			
			self.data_type = self.valobj.GetTarget().FindFirstType(map_arg_type)
			
			logger >> "and the SBType is: " + str(self.data_type)
			
			# from libstdc++ implementation of _M_root for rbtree
			self.Mroot = self.Mheader.GetChildMemberWithName('_M_parent')
			self.data_size = self.data_type.GetByteSize()
			self.skip_size = self.Mheader.GetType().GetByteSize()
		except:
			pass

	def num_children(self):
		global _map_capping_size
		logger = lldb.formatters.Logger.Logger()
		if self.count == None:
			self.count = self.num_children_impl()
			if self.count > _map_capping_size:
				self.count = _map_capping_size
		return self.count

	def num_children_impl(self):
		logger = lldb.formatters.Logger.Logger()
		try:
			root_ptr_val = self.node_ptr_value(self.Mroot)
			if root_ptr_val == 0:
				return 0;
			count = self.Mimpl.GetChildMemberWithName('_M_node_count').GetValueAsUnsigned(0)
			logger >> "I have " + str(count) + " children available"
			return count
		except:
			return 0;

	def get_child_index(self,name):
		logger = lldb.formatters.Logger.Logger()
		try:
			return int(name.lstrip('[').rstrip(']'))
		except:
			return -1

	def get_child_at_index(self,index):
		logger = lldb.formatters.Logger.Logger()
		logger >> "Being asked to fetch child[" + str(index) + "]"
		if index < 0:
			return None
		if index >= self.num_children():
			return None;
		if self.garbage:
			logger >> "Returning None since we are a garbage tree"
			return None
		try:
			offset = index
			current = self.left(self.Mheader);
			while offset > 0:
				current = self.increment_node(current)
				offset = offset - 1;
			# skip all the base stuff and get at the data
			return current.CreateChildAtOffset('['+str(index)+']',self.skip_size,self.data_type)
		except:
			return None

	# utility functions
	def node_ptr_value(self,node):
		logger = lldb.formatters.Logger.Logger()
		return node.GetValueAsUnsigned(0)

	def right(self,node):
		logger = lldb.formatters.Logger.Logger()
		return node.GetChildMemberWithName("_M_right");

	def left(self,node):
		logger = lldb.formatters.Logger.Logger()
		return node.GetChildMemberWithName("_M_left");

	def parent(self,node):
		logger = lldb.formatters.Logger.Logger()
		return node.GetChildMemberWithName("_M_parent");

	# from libstdc++ implementation of iterator for rbtree
	def increment_node(self,node):
		logger = lldb.formatters.Logger.Logger()
		max_steps = self.num_children()
		if self.node_ptr_value(self.right(node)) != 0:
			x = self.right(node);
			max_steps -= 1
			while self.node_ptr_value(self.left(x)) != 0:
				x = self.left(x);
				max_steps -= 1
				logger >> str(max_steps) + " more to go before giving up"
				if max_steps <= 0:
					self.garbage = True
					return None
			return x;
		else:
			x = node;
			y = self.parent(x)
			max_steps -= 1
			while(self.node_ptr_value(x) == self.node_ptr_value(self.right(y))):
				x = y;
				y = self.parent(y);
				max_steps -= 1
				logger >> str(max_steps) + " more to go before giving up"
				if max_steps <= 0:
					self.garbage = True
					return None
			if self.node_ptr_value(self.right(x)) != self.node_ptr_value(y):
				x = y;
			return x;

	def has_children(self):
		return self.num_children() > 0

_map_capping_size = 255
_list_capping_size = 255
_list_uses_loop_detector = True
