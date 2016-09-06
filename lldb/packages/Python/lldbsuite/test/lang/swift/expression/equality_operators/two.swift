private func == (lhs : Fooey, rhs : Fooey) -> Bool 
{ 
    Fooey.BumpCounter(2)
    return lhs.m_var != rhs.m_var // break here for two local operator
}

extension Fooey
{
    public class func CompareEm2(_ lhs : Fooey, _ rhs : Fooey) -> Bool 
    { 
        return lhs == rhs
    }
}

