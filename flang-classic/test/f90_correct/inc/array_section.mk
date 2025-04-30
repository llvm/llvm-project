#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test array_section ########


array_section: run

build:  $(SRC)/array_section.f90
	-$(RM) array_section.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/array_section.f90 -o array_section.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) array_section.$(OBJX) $(LIBS) -o array_section.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test array_section
	array_section.$(EXESUFFIX)

verify: ;

array_section.run: run

