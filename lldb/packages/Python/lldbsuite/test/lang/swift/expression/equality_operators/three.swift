import fooey

private var counter = 0

func == (lhs : Fooey, rhs : Fooey) -> Bool
{
    Fooey.BumpCounter(3)
    return lhs.m_var == rhs.m_var + 1  
}

extension Fooey
{
    class func CompareEm3(_ lhs : Fooey, _ rhs : Fooey) -> Bool
    {
        return lhs == rhs
    }
}

var lhs = Fooey()
var rhs = Fooey()

let result1 = Fooey.CompareEm1(lhs, rhs)
Fooey.ResetCounter()
let result2 = Fooey.CompareEm2(lhs, rhs)
Fooey.ResetCounter()
let result3 = Fooey.CompareEm3(lhs, rhs)
