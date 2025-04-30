#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test cm01  ########


cm01: run
	

build:  $(SRC)/cm01.f90
	-$(RM) cm01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/cm01.f90 -o cm01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) cm01.$(OBJX) $(LIBS) -o cm01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test cm01
	cm01.$(EXESUFFIX)

verify: ;

cm01.run: run

