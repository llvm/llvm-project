void pause() {}

int main()
{
  pause(); //% self.expect("repl", error=True, substrs=["Swift standard library"])
           //% self.runCmd("kill")
           //% self.expect("repl", error=True, substrs=["running process"])
  return 0;
}

