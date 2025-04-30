#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
########## Make rule for test pp73  ########


pp73: run
	

build:  $(SRC)/pp73.f90
	-$(RM) pp73.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pp73.f90 -o pp73.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pp73.$(OBJX) $(LIBS) -o pp73.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test pp73
	pp73.$(EXESUFFIX)

verify: ;

