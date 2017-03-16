add_xray_unittest(XRayBufferQueueTest SOURCES
  buffer_queue_test.cc xray_unit_test_main.cc)
add_xray_unittest(XRayFDRLoggingTest SOURCES
  fdr_logging_test.cc xray_unit_test_main.cc)

add_executable(xray_fdr_log_printer xray_fdr_log_printer_tool.cc)
