enum Enum {
    case a, b, c
}

let testA = [Enum.c: 1, Enum.b: 2]
let testB = ["a": 1, "b": 2]
let testC = (key: Enum.b, value: 2)
let testD = Set([Enum.c])
print(testA) // break here
