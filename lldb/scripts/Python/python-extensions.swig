
%extend lldb::SBAddress {
        PyObject *lldb::SBAddress::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBBlock {
        PyObject *lldb::SBBlock::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBBreakpoint {
        PyObject *lldb::SBBreakpoint::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBBreakpointLocation {
        PyObject *lldb::SBBreakpointLocation::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description, lldb::eDescriptionLevelFull);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBCommandReturnObject {
        PyObject *lldb::SBCommandReturnObject::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBCompileUnit {
        PyObject *lldb::SBCompileUnit::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBData {
        PyObject *lldb::SBData::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBDebugger {
        PyObject *lldb::SBDebugger::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBDeclaration {
        PyObject *lldb::SBDeclaration::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBError {
        PyObject *lldb::SBError::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBFileSpec {
        PyObject *lldb::SBFileSpec::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBFrame {
        PyObject *lldb::SBFrame::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBFunction {
        PyObject *lldb::SBFunction::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBInstruction {
        PyObject *lldb::SBInstruction::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBInstructionList {
        PyObject *lldb::SBInstructionList::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBLineEntry {
        PyObject *lldb::SBLineEntry::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBModule {
        PyObject *lldb::SBModule::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBProcess {
        PyObject *lldb::SBProcess::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBSection {
        PyObject *lldb::SBSection::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBSymbol {
        PyObject *lldb::SBSymbol::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBSymbolContext {
        PyObject *lldb::SBSymbolContext::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBSymbolContextList {
        PyObject *lldb::SBSymbolContextList::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBTarget {
        PyObject *lldb::SBTarget::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description, lldb::eDescriptionLevelBrief);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBType {
        PyObject *lldb::SBType::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description, lldb::eDescriptionLevelBrief);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBTypeCategory {
        PyObject *lldb::SBTypeCategory::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description, lldb::eDescriptionLevelBrief);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBTypeFilter {
        PyObject *lldb::SBTypeFilter::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description, lldb::eDescriptionLevelBrief);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBTypeFormat {
        PyObject *lldb::SBTypeFormat::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description, lldb::eDescriptionLevelBrief);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBTypeMember {
        PyObject *lldb::SBTypeMember::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description, lldb::eDescriptionLevelBrief);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBTypeNameSpecifier {
        PyObject *lldb::SBTypeNameSpecifier::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description, lldb::eDescriptionLevelBrief);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBTypeSummary {
        PyObject *lldb::SBTypeSummary::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description, lldb::eDescriptionLevelBrief);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBTypeSynthetic {
        PyObject *lldb::SBTypeSynthetic::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description, lldb::eDescriptionLevelBrief);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBThread {
        PyObject *lldb::SBThread::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBValue {
        PyObject *lldb::SBValue::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}
%extend lldb::SBValueList {
        PyObject *lldb::SBValueList::__repr__ (){
                lldb::SBStream description;
                const size_t n = $self->GetSize();
                if (n)
                {
                    for (size_t i=0; i<n; ++i)
                        $self->GetValueAtIndex(i).GetDescription(description);
                }
                else
                {
                    description.Printf("<empty> lldb.SBValueList()");
                }
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                return Py_None;
        }
}
%extend lldb::SBWatchpoint {
        PyObject *lldb::SBWatchpoint::__str__ (){
                lldb::SBStream description;
                $self->GetDescription (description, lldb::eDescriptionLevelVerbose);
                const char *desc = description.GetData();
                size_t desc_len = description.GetSize();
                if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
                    --desc_len;
                if (desc_len > 0)
                    return PyString_FromStringAndSize (desc, desc_len);
                else
                    return PyString_FromString("");
        }
}


// %extend lldb::SBDebugger {
//         // FIXME: We can't get the callback and baton
//         PyObject *lldb::SBDebugger (){
//             // Only call Py_XDECREF if we have a Python object (or NULL)
//             if (LLDBSwigPythonCallPythonLogOutputCallback == $self->GetLogOutPutCallback())
//                 Py_XDECREF($self->GetCallbackBaton());
//         }
// }
// %extend lldb::SBInputReader {
//         // FIXME: m_callback_function is private and we have no other
//         // way to access it.
//         PyObject *lldb::SBInputReader::__del__ (){
//             // Only call Py_XDECREF if we have a Python object (or NULL)
//             if (LLDBSwigPythonCallSBInputReaderCallback == $self->m_callback_function)
//                 Py_XDECREF($self->m_callback_baton);
//         }
// }

%pythoncode %{

class declaration(object):
    '''A class that represents a source declaration location with file, line and column.'''
    def __init__(self, file, line, col):
        self.file = file
        self.line = line
        self.col = col

class value(object):
    '''A class designed to wrap lldb.SBValue() objects so the resulting object
    can be used as a variable would be in code. So if you have a Point structure
    variable in your code in the current frame named "pt", you can initialize an instance
    of this class with it:
    
    pt = lldb.value(lldb.frame.FindVariable("pt"))
    print pt
    print pt.x
    print pt.y

    pt = lldb.value(lldb.frame.FindVariable("rectangle_array"))
    print rectangle_array[12]
    print rectangle_array[5].origin.x'''
    def __init__(self, sbvalue):
        self.sbvalue = sbvalue

    def __nonzero__(self):
        return self.sbvalue.__nonzero__()

    def __str__(self):
        return self.sbvalue.__str__()

    def __getitem__(self, key):
        # Allow array access if this value has children...
        if type(key) is int:
            return value(self.sbvalue.GetValueForExpressionPath("[%i]" % key))
        if type(key) is value:
            return value(self.sbvalue.GetValueForExpressionPath("[%i]" % int(key)))
        raise TypeError("No array item of type %s" % str(type(key)))

    def __getattr__(self, name):
        child_sbvalue = self.sbvalue.GetChildMemberWithName (name)
        if child_sbvalue:
            return value(child_sbvalue)
        raise AttributeError("Attribute '%s' is not defined" % name)

    def __add__(self, other):
        return int(self) + int(other)
        
    def __sub__(self, other):
        return int(self) - int(other)
        
    def __mul__(self, other):
        return int(self) * int(other)
        
    def __floordiv__(self, other):
        return int(self) // int(other)
        
    def __mod__(self, other):
        return int(self) % int(other)
        
    def __divmod__(self, other):
        return int(self) % int(other)
        
    def __pow__(self, other):
        return int(self) ** int(other)
        
    def __lshift__(self, other):
        return int(self) << int(other)
        
    def __rshift__(self, other):
        return int(self) >> int(other)
        
    def __and__(self, other):
        return int(self) & int(other)
        
    def __xor__(self, other):
        return int(self) ^ int(other)
        
    def __or__(self, other):
        return int(self) | int(other)
        
    def __div__(self, other):
        return int(self) / int(other)
        
    def __truediv__(self, other):
        return int(self) / int(other)
        
    def __iadd__(self, other):
        result = self.__add__(other)
        self.sbvalue.SetValueFromCString (str(result))
        return result
        
    def __isub__(self, other):
        result = self.__sub__(other)
        self.sbvalue.SetValueFromCString (str(result))
        return result
        
    def __imul__(self, other):
        result = self.__mul__(other)
        self.sbvalue.SetValueFromCString (str(result))
        return result
        
    def __idiv__(self, other):
        result = self.__div__(other)
        self.sbvalue.SetValueFromCString (str(result))
        return result
        
    def __itruediv__(self, other):
        result = self.__truediv__(other)
        self.sbvalue.SetValueFromCString (str(result))
        return result
        
    def __ifloordiv__(self, other):
        result =  self.__floordiv__(self, other)
        self.sbvalue.SetValueFromCString (str(result))
        return result
        
    def __imod__(self, other):
        result =  self.__and__(self, other)
        self.sbvalue.SetValueFromCString (str(result))
        return result
        
    def __ipow__(self, other):
        result = self.__pow__(self, other)
        self.sbvalue.SetValueFromCString (str(result))
        return result
        
    def __ipow__(self, other, modulo):
        result = self.__pow__(self, other, modulo)
        self.sbvalue.SetValueFromCString (str(result))
        return result
        
    def __ilshift__(self, other):
        result = self.__lshift__(other)
        self.sbvalue.SetValueFromCString (str(result))
        return result
        
    def __irshift__(self, other):
        result =  self.__rshift__(other)
        self.sbvalue.SetValueFromCString (str(result))
        return result
        
    def __iand__(self, other):
        result =  self.__and__(self, other)
        self.sbvalue.SetValueFromCString (str(result))
        return result
        
    def __ixor__(self, other):
        result =  self.__xor__(self, other)
        self.sbvalue.SetValueFromCString (str(result))
        return result
        
    def __ior__(self, other):
        result =  self.__ior__(self, other)
        self.sbvalue.SetValueFromCString (str(result))
        return result
        
    def __neg__(self):
        return -int(self)
        
    def __pos__(self):
        return +int(self)
        
    def __abs__(self):
        return abs(int(self))
        
    def __invert__(self):
        return ~int(self)
        
    def __complex__(self):
        return complex (int(self))
        
    def __int__(self):
        return self.sbvalue.GetValueAsSigned()
        
    def __long__(self):
        return self.sbvalue.GetValueAsSigned()
        
    def __float__(self):
        return float (self.sbvalue.GetValueAsSigned())
        
    def __oct__(self):
        return '0%o' % self.sbvalue.GetValueAsUnsigned()
        
    def __hex__(self):
        return '0x%x' % self.sbvalue.GetValueAsUnsigned()

    def __eq__(self, other):
        if type(other) is int:
                return int(self) == other
        elif type(other) is str:
                return str(self) == other
        elif type(other) is value:
                self_err = SBError()
                other_err = SBError()
                self_val = self.sbvalue.GetValueAsUnsigned(self_err)
                if self_err.fail:
                        raise ValueError("unable to extract value of self")
                other_val = other.sbvalue.GetValueAsUnsigned(other_err)
                if other_err.fail:
                        raise ValueError("unable to extract value of other")
                return self_val == other_val
        raise TypeError("Unknown type %s, No equality operation defined." % str(type(other)))

    def __neq__(self, other):
        return not self.__eq__(other)
%}
