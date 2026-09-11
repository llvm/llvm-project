import atexit
import logging
import argparse
import pathlib
import asyncio
import os
import sys
import signal
import transport
import protocol
from typing import Optional

logger = logging.getLogger("lldb-mcp")


class MCPClient(transport.MessageHandler):
    initialize = protocol.initialize.invoker
    initialized = protocol.initialized.invoker
    toolsList = protocol.toolsList.invoker
    toolsCall = protocol.toolsCall.invoker


def parse(uri: str) -> tuple[str, int]:
    assert uri.startswith("connection://")
    uri = uri.removeprefix("connection://")
    host, port = uri.rsplit(":", maxsplit=1)
    if host != "[::1]":
        host = host.removeprefix("[").removesuffix("]")
    return (host, int(port))


async def test_client(uri: str):
    host, port = parse(uri)
    print("connecting to", host, port)
    reader, writer = await asyncio.open_connection(host, int(port))
    with transport.Transport(reader, writer) as conn:
        async with MCPClient(conn) as client:
            _ = await client.initialize()
            client.initialized()

            tools_list_result = await client.toolsList()
            for tool in tools_list_result["tools"]:
                print("tool", tool)

            await client.toolsCall(
                name="command",
                arguments={
                    "command": "bt",
                    "debugger": "lldb://debugger/1",
                },
            )
            await client.toolsCall(
                name="debugger_list",
                arguments=None,
            )


async def launchLLDB(log_file: Optional[str] = None):
    dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(dir, "server.py")
    args = [
        "lldb",
        "-O",
        f"command script import --allow-reload {server_script}",
        "-O",
        "start_mcp" + " --log-file=" + str(log_file) if log_file else "",
    ]
    process = await asyncio.subprocess.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )

    def shutdown():
        try:
            if process.returncode is None:
                process.send_signal(signal.SIGHUP)
                os.waitpid(process.pid, 0)
        except:
            pass

    atexit.register(shutdown)


async def main() -> None:
    parser = argparse.ArgumentParser("lldb-mcp")
    parser.add_argument("-l", "--log-file", type=pathlib.Path)
    parser.add_argument("-t", "--timeout", type=float, default=30.0)
    parser.add_argument("--test", action="store_true")
    opts = parser.parse_args()
    if opts.log_file or opts.test:
        logging.basicConfig(
            filename=opts.log_file,
            format="%(created)f:%(process)d:%(levelname)s:%(name)s:%(message)s",
            level=logging.DEBUG,
        )
    logger.info("Loading lldb-mcp server configurations...")
    loop = asyncio.get_running_loop()

    launched = False
    deadline: float = loop.time() + opts.timeout
    servers: list[protocol.ServerInfo] = []
    while not servers and loop.time() < deadline:
        logger.info("loading host server details")
        servers = protocol.load()

        if not servers and not launched:
            launched = True
            logger.info("Starting lldb with server loaded...")
            await launchLLDB(log_file=opts.log_file)
            continue

        if not servers:
            logger.info("Waiting for server to start...")
            await asyncio.sleep(1.0)
            continue

        if len(servers) != 1:
            logger.error("to many lldb-mcp servers detected, exiting...")
            sys.exit(
                "Multiple servers detected, selecting a single server is not yet supported."
            )

        break

    assert servers

    if opts.test:
        for server in servers:
            await test_client(server["connection_uri"])
        return

    logger.info("Forwarding stdio to first server %r", servers[0])
    try:
        server_info = servers[0]
        host, port = parse(server_info["connection_uri"])
        cr, cw = await asyncio.open_connection(host, port)
        loop = asyncio.get_event_loop()

        def forward():
            buf = sys.stdin.buffer.read(4096)
            if not buf:  # eof detected
                cr.feed_eof()
                loop.remove_reader(sys.stdin)
                return
            logger.info("--> %s", buf.decode().strip())
            cw.write(buf)

        os.set_blocking(sys.stdin.fileno(), False)
        loop.add_reader(sys.stdin, forward)
        async for f in cr:
            logger.info("<-- %s", f.decode().strip())
            sys.stdout.buffer.write(f)
            sys.stdout.buffer.flush()
    except:
        logger.exception("forwarding client failed")
    finally:
        logger.info("lldb-mcp client shut down")


if __name__ == "__main__":
    asyncio.run(main())
