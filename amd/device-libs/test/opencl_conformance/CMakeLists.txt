################################################################################
##
## The University of Illinois/NCSA
## Open Source License (NCSA)
##
## Copyright (c) 2016, Advanced Micro Devices, Inc. All rights reserved.
##
## Developed by:
##
##                 AMD Research and AMD HSA Software Development
##
##                 Advanced Micro Devices, Inc.
##
##                 www.amd.com
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal with the Software without restriction, including without limitation
## the rights to use, copy, modify, merge, publish, distribute, sublicense,
## and#or sell copies of the Software, and to permit persons to whom the
## Software is furnished to do so, subject to the following conditions:
##
##  - Redistributions of source code must retain the above copyright notice,
##    this list of conditions and the following disclaimers.
##  - Redistributions in binary form must reproduce the above copyright
##    notice, this list of conditions and the following disclaimers in
##    the documentation and#or other materials provided with the distribution.
##  - Neither the names of Advanced Micro Devices, Inc,
##    nor the names of its contributors may be used to endorse or promote
##    products derived from this Software without specific prior written
##    permission.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
## THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
## OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
## ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
## DEALINGS WITH THE SOFTWARE.
##
################################################################################

if(OCL_CONFORMANCE_HOME)
  file(GLOB files
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_0.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_1.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_10.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_100.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_101.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_102.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_103.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_104.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_105.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_106.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_107.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_108.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_109.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_11.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_110.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_111.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_112.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_113.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_114.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_115.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_116.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_117.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_118.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_119.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_12.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_120.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_121.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_122.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_123.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_124.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_125.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_126.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_127.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_128.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_129.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_13.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_130.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_131.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_132.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_133.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_134.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_135.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_136.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_137.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_138.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_139.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_14.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_140.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_141.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_142.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_143.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_144.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_145.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_146.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_147.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_148.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_149.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_15.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_150.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_151.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_152.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_153.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_16.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_17.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_18.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_19.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_2.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_20.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_21.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_22.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_23.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_24.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_25.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_26.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_27.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_28.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_29.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_3.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_30.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_31.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_32.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_33.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_34.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_35.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_36.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_37.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_38.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_39.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_4.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_40.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_41.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_42.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_43.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_44.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_45.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_46.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_47.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_48.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_49.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_5.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_50.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_51.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_52.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_53.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_54.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_55.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_56.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_57.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_58.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_59.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_6.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_60.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_61.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_62.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_63.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_64.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_65.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_66.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_67.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_68.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_69.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_7.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_70.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_71.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_72.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_73.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_74.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_75.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_76.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_77.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_78.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_79.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_8.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_80.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_81.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_82.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_83.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_84.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_85.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_86.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_87.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_88.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_89.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_9.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_90.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_91.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_92.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_93.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_94.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_95.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_96.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_97.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_98.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/commonfns_99.cl
  )
  foreach(file ${files})
    file(RELATIVE_PATH rfile ${OCL_CONFORMANCE_HOME} ${file})
    clang_opencl_test_file(${OCL_CONFORMANCE_HOME} ${rfile})
  endforeach()
endif()
