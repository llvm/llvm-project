#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test format_specification  ########


format_specification: run


build:  $(SRC)/format_specification.f90
	-$(RM) format_specification.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/format_specification.f90 -o format_specification.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) format_specification.$(OBJX) -o format_specification.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test format_specification
	format_specification.$(EXESUFFIX)

verify: ;

format_specification.run: run

