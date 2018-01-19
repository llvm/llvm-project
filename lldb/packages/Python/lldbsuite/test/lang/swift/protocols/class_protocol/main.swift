import Foundation

protocol HasWhy : class
{
  var why : Int { get }
}

class ShouldBeWhy : NSObject, HasWhy
{
  var why : Int = 10
}

class ByWhy<T> where T : HasWhy 
{
  let myWhy : Int
  init(input : T)
  {
    myWhy = input.why  // Break here and print input
  }
}

func doIt() 
{
  let mySBW = ShouldBeWhy()
  let myByWhy = ByWhy(input: mySBW)
  print(myByWhy.myWhy)
}

doIt()