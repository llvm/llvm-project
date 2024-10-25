import os
import sys
import tempfile

if __name__ == "__main__":
  while True:
    fifo_path = sys.argv[1]
    with open(sys.argv[1], 'rb') as fifo:
      filename, ext = os.path.splitext(os.path.basename(fifo_path))
      fd, _ = tempfile.mkstemp(
                suffix=ext, prefix=f"{filename}.", dir=os.path.dirname(fifo_path))
      with os.fdopen(fd, 'wb') as out:
        out.write(fifo.read())

