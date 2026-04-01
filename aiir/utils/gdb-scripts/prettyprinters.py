"""GDB pretty printers for AIIR types."""

import gdb.printing


class StoragePrinter:
    """Prints bases of a struct and its fields."""

    def __init__(self, val):
        self.val = val

    def children(self):
        for field in self.val.type.fields():
            if field.is_base_class:
                yield "<%s>" % field.name, self.val.cast(field.type)
            else:
                yield field.name, self.val[field.name]

    def to_string(self):
        return "aiir::Storage"


class TupleTypeStoragePrinter(StoragePrinter):
    def children(self):
        for child in StoragePrinter.children(self):
            yield child
        pointer_type = gdb.lookup_type("aiir::Type").pointer()
        elements = (self.val.address + 1).cast(pointer_type)
        for i in range(self.val["numElements"]):
            yield "elements[%u]" % i, elements[i]

    def to_string(self):
        return "aiir::TupleTypeStorage of %u elements" % self.val["numElements"]


class FusedLocationStoragePrinter(StoragePrinter):
    def children(self):
        for child in StoragePrinter.children(self):
            yield child
        pointer_type = gdb.lookup_type("aiir::Location").pointer()
        elements = (self.val.address + 1).cast(pointer_type)
        for i in range(self.val["numLocs"]):
            yield "locs[%u]" % i, elements[i]

    def to_string(self):
        return "aiir::FusedLocationStorage of %u locs" % self.val["numLocs"]


class StorageTypeMap:
    """Maps a TypeID to the corresponding concrete type.

    Types need to be registered by name before the first lookup.
    """

    def __init__(self):
        self.map = None
        self.type_names = []

    def register_type(self, type_name):
        assert not self.map, "register_type called after __getitem__"
        self.type_names += [type_name]

    def _init_map(self):
        """Lazy initialization  of self.map."""
        if self.map:
            return
        self.map = {}
        for type_name in self.type_names:
            concrete_type = gdb.lookup_type(type_name)
            try:
                storage = gdb.parse_and_eval(
                    "&'aiir::detail::TypeIDExported::get<%s>()::instance'" % type_name
                )
            except gdb.error:
                # Skip when TypeID instance cannot be found in current context.
                continue
            if concrete_type and storage:
                self.map[int(storage)] = concrete_type

    def __getitem__(self, type_id):
        self._init_map()
        return self.map.get(int(type_id["storage"]))


storage_type_map = StorageTypeMap()


def get_type_id_printer(val):
    """Returns a printer of the name of a aiir::TypeID."""

    class TypeIdPrinter:
        def __init__(self, string):
            self.string = string

        def to_string(self):
            return self.string

    concrete_type = storage_type_map[val]
    if not concrete_type:
        return None
    return TypeIdPrinter("aiir::TypeID::get<%s>()" % concrete_type)


def get_attr_or_type_printer(val, get_type_id):
    """Returns a printer for aiir::Attribute or aiir::Type."""

    class AttrOrTypePrinter:
        def __init__(self, type_id, impl):
            self.type_id = type_id
            self.impl = impl

        def children(self):
            yield "typeID", self.type_id
            yield "impl", self.impl

        def to_string(self):
            return "cast<%s>" % self.impl.type

    if not val["impl"]:
        return None
    impl = val["impl"].dereference()
    type_id = get_type_id(impl)
    concrete_type = storage_type_map[type_id]
    if not concrete_type:
        return None
    # 3rd template argument of StorageUserBase is the storage type.
    storage_type = concrete_type.fields()[0].type.template_argument(2)
    if not storage_type:
        return None
    return AttrOrTypePrinter(type_id, impl.cast(storage_type))


class ImplPrinter:
    """Printer for an instance with a single 'impl' member pointer."""

    def __init__(self, val):
        self.val = val
        self.impl = val["impl"]

    def children(self):
        if self.impl:
            yield "impl", self.impl.dereference()

    def to_string(self):
        return self.val.type.name


# Printers of types deriving from Attribute::AttrBase or Type::TypeBase.
for name in [
    # aiir/IR/Attributes.h
    "ArrayAttr",
    "DictionaryAttr",
    "FloatAttr",
    "IntegerAttr",
    "IntegerSetAttr",
    "OpaqueAttr",
    "StringAttr",
    "SymbolRefAttr",
    "TypeAttr",
    "UnitAttr",
    "DenseStringElementsAttr",
    "DenseTypedElementsAttr",
    "SparseElementsAttr",
    # aiir/IR/BuiltinTypes.h
    "ComplexType",
    "IndexType",
    "IntegerType",
    "Float16Type",
    "FloatTF32Type",
    "Float32Type",
    "Float64Type",
    "Float80Type",
    "Float128Type",
    "NoneType",
    "VectorType",
    "RankedTensorType",
    "UnrankedTensorType",
    "MemRefType",
    "UnrankedMemRefType",
    "TupleType",
    # aiir/IR/Location.h
    "CallSiteLoc",
    "FileLineColLoc",
    "FusedLoc",
    "NameLoc",
    "OpaqueLoc",
    "UnknownLoc",
]:
    storage_type_map.register_type("aiir::%s" % name)  # Register for upcasting.
storage_type_map.register_type("void")  # Register default.


pp = gdb.printing.RegexpCollectionPrettyPrinter("AIIRSupport")

pp.add_printer("aiir::OperationName", "^aiir::OperationName$", ImplPrinter)
pp.add_printer("aiir::Value", "^aiir::Value$", ImplPrinter)

# Printers for types deriving from AttributeStorage or TypeStorage.
pp.add_printer(
    "aiir::detail::FusedLocationStorage",
    "^aiir::detail::FusedLocationStorage",
    FusedLocationStoragePrinter,
)
pp.add_printer(
    "aiir::detail::TupleTypeStorage",
    "^aiir::detail::TupleTypeStorage$",
    TupleTypeStoragePrinter,
)

pp.add_printer("aiir::TypeID", "^aiir::TypeID$", get_type_id_printer)


def add_attr_or_type_printers(name):
    """Adds printers for aiir::Attribute or aiir::Type and their Storage type."""
    get_type_id = lambda val: val["abstract%s" % name]["typeID"]
    pp.add_printer(
        "aiir::%s" % name,
        "^aiir::%s$" % name,
        lambda val: get_attr_or_type_printer(val, get_type_id),
    )


# Upcasting printers of aiir::Attribute and aiir::Type.
for name in ["Attribute", "Type"]:
    add_attr_or_type_printers(name)

gdb.printing.register_pretty_printer(gdb.current_objfile(), pp)
