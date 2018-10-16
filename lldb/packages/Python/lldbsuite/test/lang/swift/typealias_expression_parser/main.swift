func main() -> Int {
  typealias Pair<T> = (T, T)
  let patatino : Pair<Int> = (1, 2)
  return 0 //%self.expect('expr -d run -- patatino', substrs=['Pair<Int>'])
}

main()
