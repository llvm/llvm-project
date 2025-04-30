#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test iso003  ########


iso003: run
	

build:  $(SRC)/iso003.f90
	-$(RM) iso003.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) $(FFLAGS) $(LDFLAGS) $(SRC)/iso003.f90 -o iso003.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test iso003
	iso003.$(EXESUFFIX)

verify: ;

iso003.run: run

