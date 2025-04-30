#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
########## Make rule for test en01  ########


en01: run
	

build:  $(SRC)/en01.f90
	-$(RM) en01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/en01.f90 -o en01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) en01.$(OBJX) $(LIBS) -o en01.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test en01
	en01.$(EXESUFFIX)

verify: ;

