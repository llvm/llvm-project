# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# This test checks whether the largest possible floating-point number works.
build:
	@echo ------------------------------------- building test $@
	$(FC) $(FFLAGS) $(SRC)/$(TEST).f08 -o $(TEST).$(EXE)

run:
	@echo ------------------------------------ executing test $@
	./$(TEST).$(EXE)

verify:
	@echo ------------------------------------ verifying
	@echo test should have printed verification above

