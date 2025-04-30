#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test iso002  ########


iso002: run
	

build:  $(SRC)/iso002.f90
	-$(RM) iso002.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) $(FFLAGS) $(LDFLAGS) $(SRC)/iso002.f90 -o iso002.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test iso002
	iso002.$(EXESUFFIX)

verify: ;

iso002.run: run

