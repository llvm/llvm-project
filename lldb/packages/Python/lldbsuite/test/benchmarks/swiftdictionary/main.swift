func main() -> Int {
    var dict = [Int: Int]()
    for i in 0..<1500 {
        dict[i] = i
    }
    return dict.count // break here
}

print(main())
