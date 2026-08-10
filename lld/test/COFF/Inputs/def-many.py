import sys

print("EXPORTS")
for i in range(0, int(sys.argv[1])):
  print("f%d=f" % (i))
