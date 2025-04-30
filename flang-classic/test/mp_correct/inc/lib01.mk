#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
lib01: lib01.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN4) ./a.$(EXESUFFIX) $(LOG)
lib01.$(OBJX): $(SRC)/lib01.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/lib01.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) lib01.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: lib01
run: ;
