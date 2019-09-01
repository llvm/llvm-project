class Issue11581SyntheticProvider(object):

    def __init__(self, valobj, dict):
        self.valobj = valobj
        self.addrOf = valobj.AddressOf()
        self.addr = valobj.GetAddress()
        self.load_address = valobj.GetLoadAddress()

    def num_children(self):
        return 3

    def get_child_at_index(self, index):
        if index == 0:
            return self.addrOf
        if index == 1:
            return self.valobj.CreateValueFromExpression(
                "addr", str(self.addr))
        if index == 2:
            return self.valobj.CreateValueFromExpression(
                "load_address", str(self.load_address))

    def get_child_index(self, name):
        if name == "addrOf":
            return 0
        if name == "addr":
            return 1
        if name == "load_address":
            return 2
