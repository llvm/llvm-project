When performing a code review, pay close attention to code modifying a function
control flow. Could the change result in the corruption of performance profile
data? Could the change result in invalid debug information, in particular for
branches and calls?
