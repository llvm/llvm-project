const path = process.env.GITHUB_WORKSPACE + '/*';
console.log('Cleaning ' + path)
require('child_process').execSync('rm -rf ' + path);
require('child_process').execSync('rm -rf ' + process.env.GITHUB_WORKSPACE + '/.git');
