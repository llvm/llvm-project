from __future__ import print_function


import gdbremote_testcase
import lldbgdbserverutils
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

import json

class TestAppleSimulatorOSType(gdbremote_testcase.GdbRemoteTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.skipIfOutOfTreeDebugserver
    def check_simulator_ostype(self, sdk, platform, arch='x86_64'):
        sim_devices_str = subprocess.check_output(['xcrun', 'simctl', 'list',
                                                   '-j', 'devices']).decode("utf-8")
        sim_devices = json.loads(sim_devices_str)['devices']
        # Find an available simulator for the requested platform
        deviceUDID = None
        for simulator in sim_devices:
            if isinstance(simulator,dict):
                runtime = simulator['name']
                devices = simulator['devices']
            else:
                runtime = simulator
                devices = sim_devices[simulator]
            if not platform in runtime.lower():
                continue
            for device in devices:
                if device['availability'] != '(available)':
                    continue
                deviceUDID = device['udid']
                break
            if deviceUDID != None:
                break

        # Launch the process using simctl
        self.assertIsNotNone(deviceUDID)
        exe_name = 'test_simulator_platform_{}'.format(platform)
        sdkroot = subprocess.check_output(['xcrun', '--show-sdk-path', '--sdk',
                                           sdk]).decode("utf-8")
        self.build(dictionary={ 'EXE': exe_name, 'SDKROOT': sdkroot.strip(),
                                'ARCH': arch })
        exe_path = self.getBuildArtifact(exe_name)
        sim_launcher = subprocess.Popen(['xcrun', 'simctl', 'spawn',
                                         deviceUDID, exe_path,
                                         'print-pid', 'sleep:10'],
                                        stderr=subprocess.PIPE)
        # Get the PID from the process output
        pid = None
        while not pid:
            stderr = sim_launcher.stderr.readline().decode("utf-8")
            if stderr == '':
                continue
            m = re.match(r"PID: (.*)", stderr)
            self.assertIsNotNone(m)
            pid = int(m.group(1))

        # Launch debug monitor attaching to the simulated process
        self.init_debugserver_test()
        server = self.connect_to_debug_monitor(attach_pid=pid)

        # Setup packet sequences
        self.add_no_ack_remote_stream()
        self.add_process_info_collection_packets()
        self.test_sequence.add_log_lines(
            ["read packet: " +
             "$jGetLoadedDynamicLibrariesInfos:{\"fetch_all_solibs\" : true}]#ce",
             {"direction": "send", "regex": r"^\$(.+)#[0-9a-fA-F]{2}$",
              "capture": {1: "dylib_info_raw"}}],
            True)

        # Run the stream
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather process info response
        process_info = self.parse_process_info_response(context)
        self.assertIsNotNone(process_info)

        # Check that ostype is correct
        self.assertEquals(process_info['ostype'], platform)

        # Now for dylibs
        dylib_info_raw = context.get("dylib_info_raw")
        dylib_info = json.loads(self.decode_gdbremote_binary(dylib_info_raw))
        images = dylib_info['images']

        image_info = None
        for image in images:
            if image['pathname'] != exe_path:
                continue
            image_info = image
            break

        self.assertIsNotNone(image_info)
        self.assertEquals(image['min_version_os_name'], platform)


    @apple_simulator_test('iphone')
    @debugserver_test
    @skipIfDarwinEmbedded
    def test_simulator_ostype_ios(self):
        self.check_simulator_ostype(sdk='iphonesimulator',
                                    platform='ios')

    @apple_simulator_test('appletv')
    @debugserver_test
    @skipIfDarwinEmbedded
    def test_simulator_ostype_tvos(self):
        self.check_simulator_ostype(sdk='appletvsimulator',
                                    platform='tvos')

    @apple_simulator_test('watch')
    @debugserver_test
    @skipIfDarwinEmbedded
    def test_simulator_ostype_watchos(self):
        self.check_simulator_ostype(sdk='watchsimulator',
                                    platform='watchos', arch='i386')
