
from __future__ import print_function


import re
import select
import threading
import traceback
import codecs

from six.moves import queue


def _handle_output_packet_string(packet_contents):
    if (not packet_contents) or (len(packet_contents) < 1):
        return None
    elif packet_contents[0] != "O":
        return None
    elif packet_contents == "OK":
        return None
    else:
        return packet_contents[1:].decode("hex")


def _dump_queue(the_queue):
    while not the_queue.empty():
        print(codecs.encode(the_queue.get(True), "string_escape"))
        print("\n")


class PumpQueues(object):

    def __init__(self):
        self._output_queue = queue.Queue()
        self._packet_queue = queue.Queue()

    def output_queue(self):
        return self._output_queue

    def packet_queue(self):
        return self._packet_queue

    def verify_queues_empty(self):
        # Warn if there is any content left in any of the queues.
        # That would represent unmatched packets.
        if not self.output_queue().empty():
            print("warning: output queue entries still exist:")
            _dump_queue(self.output_queue())
            print("from here:")
            traceback.print_stack()

        if not self.packet_queue().empty():
            print("warning: packet queue entries still exist:")
            _dump_queue(self.packet_queue())
            print("from here:")
            traceback.print_stack()


class SocketPacketPump(object):
    """A threaded packet reader that partitions packets into two streams.

    All incoming $O packet content is accumulated with the current accumulation
    state put into the OutputQueue.

    All other incoming packets are placed in the packet queue.

    A select thread can be started and stopped, and runs to place packet
    content into the two queues.
    """

    _GDB_REMOTE_PACKET_REGEX = re.compile(r'^\$([^\#]*)#[0-9a-fA-F]{2}')

    def __init__(self, pump_socket, pump_queues, logger=None):
        if not pump_socket:
            raise Exception("pump_socket cannot be None")

        self._thread = None
        self._stop_thread = False
        self._socket = pump_socket
        self._logger = logger
        self._receive_buffer = ""
        self._accumulated_output = ""
        self._pump_queues = pump_queues

    def __enter__(self):
        """Support the python 'with' statement.

        Start the pump thread."""
        self.start_pump_thread()
        return self

    def __exit__(self, exit_type, value, the_traceback):
        """Support the python 'with' statement.

        Shut down the pump thread."""
        self.stop_pump_thread()

    def start_pump_thread(self):
        if self._thread:
            raise Exception("pump thread is already running")
        self._stop_thread = False
        self._thread = threading.Thread(target=self._run_method)
        self._thread.start()

    def stop_pump_thread(self):
        self._stop_thread = True
        if self._thread:
            self._thread.join()

    def _process_new_bytes(self, new_bytes):
        if not new_bytes:
            return
        if len(new_bytes) < 1:
            return

        # Add new bytes to our accumulated unprocessed packet bytes.
        self._receive_buffer += new_bytes

        # Parse fully-formed packets into individual packets.
        has_more = len(self._receive_buffer) > 0
        while has_more:
            if len(self._receive_buffer) <= 0:
                has_more = False
            # handle '+' ack
            elif self._receive_buffer[0] == "+":
                self._pump_queues.packet_queue().put("+")
                self._receive_buffer = self._receive_buffer[1:]
                if self._logger:
                    self._logger.debug(
                        "parsed packet from stub: +\n" +
                        "new receive_buffer: {}".format(
                            self._receive_buffer))
            else:
                packet_match = self._GDB_REMOTE_PACKET_REGEX.match(
                    self._receive_buffer)
                if packet_match:
                    # Our receive buffer matches a packet at the
                    # start of the receive buffer.
                    new_output_content = _handle_output_packet_string(
                        packet_match.group(1))
                    if new_output_content:
                        # This was an $O packet with new content.
                        self._accumulated_output += new_output_content
                        self._pump_queues.output_queue().put(self._accumulated_output)
                    else:
                        # Any packet other than $O.
                        self._pump_queues.packet_queue().put(packet_match.group(0))

                    # Remove the parsed packet from the receive
                    # buffer.
                    self._receive_buffer = self._receive_buffer[
                        len(packet_match.group(0)):]
                    if self._logger:
                        self._logger.debug(
                            "parsed packet from stub: " +
                            packet_match.group(0))
                        self._logger.debug(
                            "new receive_buffer: " +
                            self._receive_buffer)
                else:
                    # We don't have enough in the receive bufferto make a full
                    # packet. Stop trying until we read more.
                    has_more = False

    def _run_method(self):
        self._receive_buffer = ""
        self._accumulated_output = ""

        if self._logger:
            self._logger.info("socket pump starting")

        # Keep looping around until we're asked to stop the thread.
        while not self._stop_thread:
            can_read, _, _ = select.select([self._socket], [], [], 0)
            if can_read and self._socket in can_read:
                try:
                    new_bytes = self._socket.recv(4096)
                    if self._logger and new_bytes and len(new_bytes) > 0:
                        self._logger.debug(
                            "pump received bytes: {}".format(new_bytes))
                except:
                    # Likely a closed socket.  Done with the pump thread.
                    if self._logger:
                        self._logger.debug(
                            "socket read failed, stopping pump read thread\n" +
                            traceback.format_exc(3))
                    break
                self._process_new_bytes(new_bytes)

        if self._logger:
            self._logger.info("socket pump exiting")

    def get_accumulated_output(self):
        return self._accumulated_output

    def get_receive_buffer(self):
        return self._receive_buffer
