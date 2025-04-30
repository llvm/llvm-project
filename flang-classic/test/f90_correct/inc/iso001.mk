#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test iso001  ########


iso001: run
	

build:  $(SRC)/iso001.f90
	-$(RM) iso001.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) $(FFLAGS) $(LDFLAGS) $(SRC)/iso001.f90 -o iso001.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test iso001
	iso001.$(EXESUFFIX)

verify: ;

iso001.run: run

